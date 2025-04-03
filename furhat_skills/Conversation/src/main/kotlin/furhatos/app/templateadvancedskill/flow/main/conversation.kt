package furhatos.app.templateadvancedskill.flow.main

import furhatos.flow.kotlin.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.net.ConnectException
import okio.IOException
import furhatos.nlu.common.*
import furhatos.app.templateadvancedskill.flow.Parent
import furhatos.gestures.Gestures
import furhatos.app.templateadvancedskill.params.LOCAL_BACKEND_URL
import furhatos.app.templateadvancedskill.params.AWS_BACKEND_URL
import java.util.concurrent.TimeUnit
import furhatos.app.templateadvancedskill.nlu.UncertainResponseIntent
import java.net.SocketTimeoutException

data class Transcription(val content: String)

data class EngageRequest(val document: String, val answer: String)

// Document Q&A state, inheriting from Parent.
fun documentInfoQnA(documentName: String): State = state(parent = Parent) {
    var conversationCount = 0
    var lastQuestion = ""
    var lastAnswer = ""
    var previousQuestions = mutableListOf<String>()
    var previousAnswers = mutableListOf<String>()
    var userMood = "neutral" // Track user's mood
    var lastGestureTime = 0L

    onEntry {
        // Lock the attended user during this conversation.
        furhat.gesture(Gestures.Smile)
        furhat.ask("Hello! I'm here to help you learn about $documentName. What would you like to know?")
    }

    onExit {
        // Release the attention lock when leaving this state.
        furhat.gesture(Gestures.Wink)
    }

    onResponse<Goodbye> {
        furhat.gesture(Gestures.Smile)
        furhat.say("Thank you for the interesting conversation! Goodbye!")
        goto(Idle)
    }

    onResponse<No> {
        // Only end conversation if it's a clear "no" without additional context
        if (it.text.matches(Regex("(?i)^(no|nope|nah)$"))) {
            furhat.gesture(Gestures.Nod)
            furhat.say("Alright, thank you for the conversation. Goodbye!")
            goto(Idle)
        } else {
            // If it's a "no" with additional context, treat it as a regular response
            raise(it)
        }
    }

    onResponse<UncertainResponseIntent> {
        // Handle uncertain responses by encouraging further discussion
        furhat.gesture(Gestures.Thoughtful)
        furhat.say {
            random {
                +"That's an interesting perspective. Let me share what I know about this topic."
                +"I understand your uncertainty. Let me provide some more information that might help."
                +"That's a good point to explore further. Let me elaborate on this topic."
            }
        }
        // Add an engaging follow-up question
        furhat.ask {
            random {
                +"What specific aspect of this topic interests you the most?"
                +"Would you like to explore a particular angle of this discussion?"
                +"Is there a specific part you'd like me to focus on?"
            }
        }
    }

    onResponse {
        val userQuestion = it.text.trim()
        conversationCount++
        
        // Update user mood based on question content
        userMood = when {
            userQuestion.contains(Regex("(great|wonderful|amazing|excellent)", RegexOption.IGNORE_CASE)) -> "positive"
            userQuestion.contains(Regex("(bad|terrible|awful|horrible)", RegexOption.IGNORE_CASE)) -> "negative"
            else -> "neutral"
        }
        
        // Natural thinking gesture - only if the question is complex
        if (userQuestion.split(" ").size > 5) {
            furhat.gesture(Gestures.GazeAway, priority = 1)
        }
        
        // Call the backend /ask endpoint to get an answer
        val answer = callDocumentAgent(userQuestion)
        
        // Clean up the answer - remove URLs and extra whitespace, but keep the complete response
        val cleanAnswer = answer
            .replace(Regex("https?://\\S+"), "")
            .replace(Regex("\\s+"), " ")
            .trim()
        
        // Store the Q&A pair for context
        previousQuestions.add(userQuestion)
        previousAnswers.add(cleanAnswer)
        lastQuestion = userQuestion
        lastAnswer = cleanAnswer
        
        // Add natural gestures while speaking, but less frequently
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastGestureTime > 5000) { // Increased to 5 seconds between gestures
            when (userMood) {
                "positive" -> furhat.gesture(Gestures.Smile, priority = 2)
                "negative" -> furhat.gesture(Gestures.ExpressSad, priority = 2)
                else -> furhat.gesture(Gestures.Nod, priority = 2)
            }
            lastGestureTime = currentTime
        }
        
        // Speak the cleaned answer with natural pauses
        furhat.say(cleanAnswer)
        
        // Generate a contextual follow-up based on conversation history and user mood
        val followUpPrompt = when {
            conversationCount == 1 -> {
                // First follow-up: Focus on specific aspects mentioned in the answer
                val keyAspects = cleanAnswer.split(".").take(2).joinToString(" ")
                when (userMood) {
                    "positive" -> "What would you like to know more about?"
                    "negative" -> "Would you like me to explain that differently?"
                    else -> "What interests you most about that?"
                }
            }
            conversationCount == 2 -> {
                // Second follow-up: Connect to previous question
                when (userMood) {
                    "positive" -> "Want to explore that further?"
                    "negative" -> "Would you like me to clarify anything?"
                    else -> "What would you like to know more about?"
                }
            }
            else -> {
                // Later follow-ups: Use conversation history for context
                try {
                    val engagePrompt = callEngageUser(documentName, cleanAnswer)
                    if (engagePrompt.isNotEmpty()) {
                        // Keep the API response simple and natural
                        engagePrompt
                    } else {
                        // Simple fallback based on mood
                        when (userMood) {
                            "positive" -> "What would you like to explore next?"
                            "negative" -> "Would you like me to explain something else?"
                            else -> "What interests you most?"
                        }
                    }
                } catch (e: Exception) {
                    // Simple fallback based on mood
                    when (userMood) {
                        "positive" -> "What would you like to explore next?"
                        "negative" -> "Would you like me to explain something else?"
                        else -> "What interests you most?"
                    }
                }
            }
        }
        
        // Ask the follow-up question with appropriate gesture
        when (userMood) {
            "positive" -> furhat.gesture(Gestures.Smile, priority = 2)
            "negative" -> furhat.gesture(Gestures.ExpressSad, priority = 2)
            else -> furhat.gesture(Gestures.Nod, priority = 2)
        }
        furhat.ask(followUpPrompt)
    }

    onNoResponse {
        furhat.gesture(Gestures.ExpressSad)
        furhat.ask("I didn't catch that. Could you please repeat your question?")
        reentry()
    }
}

// Helper function to call the /ask endpoint.
private fun callDocumentAgent(question: String): String {
    val baseUrl = AWS_BACKEND_URL
    val client = OkHttpClient.Builder()
        .connectTimeout(60, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()
    return try {
        val requestBody = JSONObject()
            .put("content", question)
            .put("max_tokens", 2000)  // Increased token limit
            .put("temperature", 0.7)  // Add temperature for more controlled generation
            .put("top_p", 0.9)        // Add top_p for better response quality
            .toString()
            .toRequestBody("application/json; charset=utf-8".toMediaType())
        
        val request = Request.Builder()
            .url("$baseUrl/ask")
            .post(requestBody)
            .build()

        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                println("Error response from backend: ${response.code} - ${response.message}")
                throw IOException("Unexpected response: $response")
            }
            
            val jsonResponse = response.body?.string() ?: throw IOException("Empty response")
            val jsonObject = JSONObject(jsonResponse)
            
            // Log the response length for debugging
            val responseText = jsonObject.getString("response")
            println("Response length: ${responseText.length} characters")
            
            // Check for potential truncation
            if (responseText.endsWith("...") || 
                responseText.endsWith(".") == false || 
                responseText.length > 1900) {  // Close to max_tokens limit
                println("Warning: Response might be truncated")
                // You might want to handle this case differently, e.g., by requesting continuation
            }
            
            responseText
        }
    } catch (e: ConnectException) {
        println("Connection error: ${e.message}")
        "I'm sorry, I cannot process your request right now. Please try again in a moment."
    } catch (e: SocketTimeoutException) {
        println("Timeout error: ${e.message}")
        "I'm sorry, the request took too long to process. Please try asking your question again."
    } catch (e: Exception) {
        println("Error processing question: ${e.message}")
        "I apologize, but I encountered an error processing your question. Could you please rephrase it?"
    }
}

// Helper function to call the /engage endpoint.
private fun callEngageUser(documentName: String, answer: String): String {
    val baseUrl = AWS_BACKEND_URL
    val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()
    return try {
        val map = JSONObject()
        map.put("document", documentName)
        map.put("answer", answer)
        val requestBody = map.toString()
            .toRequestBody("application/json; charset=utf-8".toMediaType())

        val request = Request.Builder()
            .url("$baseUrl/engage")
            .post(requestBody)
            .build()

        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) throw IOException("Unexpected response: $response")
            val jsonResponse = response.body?.string() ?: throw IOException("Empty response")
            val jsonObject = JSONObject(jsonResponse)
            try {
                jsonObject.getString("prompt")
            } catch (e: Exception) {
                ""  // Return empty string to trigger fallback question
            }
        }
    } catch (e: ConnectException) {
        ""  // Return empty string to trigger fallback question
    } catch (e: Exception) {
        ""  // Return empty string to trigger fallback question
    }
}