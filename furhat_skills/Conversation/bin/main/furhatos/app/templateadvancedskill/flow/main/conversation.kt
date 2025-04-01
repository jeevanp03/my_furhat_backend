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
        furhat.gesture(Gestures.Nod)
        furhat.say("Alright, thank you for the conversation. Goodbye!")
        goto(Idle)
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
        
        // Clean up the answer
        val cleanAnswer = answer.split("Q1")[0]
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
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()
    return try {
        val requestBody = JSONObject().put("content", question).toString()
            .toRequestBody("application/json; charset=utf-8".toMediaType())
        val request = Request.Builder()
            .url("$baseUrl/ask")
            .post(requestBody)
            .build()

        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) throw IOException("Unexpected response: $response")
            val jsonResponse = response.body?.string() ?: throw IOException("Empty response")
            val jsonObject = JSONObject(jsonResponse)
            jsonObject.getString("response")
        }
    } catch (e: ConnectException) {
        "I'm sorry, I cannot process your request right now."
    } catch (e: Exception) {
        "I apologize, but I encountered an error processing your question."
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