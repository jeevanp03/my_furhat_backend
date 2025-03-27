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
        furhat.say("Hello! I'm here to help you learn about $documentName. What would you like to know?")
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
        
        // Signal the robot is thinking with natural gestures
        furhat.gesture(Gestures.GazeAway)
        furhat.say("Let me think about that...")
        
        // Call the backend /ask endpoint to get an answer.
        val answer = callDocumentAgent(userQuestion)
        
        // Clean up the answer by removing Q&A sections and URLs
        val cleanAnswer = answer.split("Q1")[0]  // Take only the first part before any Q&A
            .replace(Regex("https?://\\S+"), "")  // Remove URLs
            .replace(Regex("\\s+"), " ")  // Normalize whitespace
            .trim()
        
        // Store the Q&A pair for context
        previousQuestions.add(userQuestion)
        previousAnswers.add(cleanAnswer)
        lastQuestion = userQuestion
        lastAnswer = cleanAnswer
        
        // Add natural gestures while speaking
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastGestureTime > 3000) { // Minimum 3 seconds between gestures
            when (userMood) {
                "positive" -> {
                    furhat.gesture(Gestures.Smile)
                    furhat.gesture(Gestures.Nod)
                }
                "negative" -> {
                    furhat.gesture(Gestures.ExpressSad)
                    furhat.gesture(Gestures.Shake)
                }
                else -> {
                    furhat.gesture(Gestures.Nod)
                    furhat.gesture(Gestures.GazeAway)
                }
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
                    "positive" -> "I'm glad you're interested in $keyAspects! What would you like to know more about that?"
                    "negative" -> "I understand this might be challenging. Let me explain more about $keyAspects. What would you like to know?"
                    else -> "I notice you're interested in $keyAspects. What would you like to know more about that?"
                }
            }
            conversationCount == 2 -> {
                // Second follow-up: Connect to previous question
                val previousTopic = previousQuestions[0].split(" ").take(3).joinToString(" ")
                when (userMood) {
                    "positive" -> "You asked about $previousTopic earlier. I'd love to explore how that connects to what we just discussed!"
                    "negative" -> "You asked about $previousTopic earlier. Let me clarify how that relates to what we just discussed."
                    else -> "You asked about $previousTopic earlier. Would you like to explore how that relates to what we just discussed?"
                }
            }
            else -> {
                // Later follow-ups: Use conversation history for context
                try {
                    // Try to get an engaging follow-up from the API
                    val engagePrompt = callEngageUser(documentName, cleanAnswer)
                    if (engagePrompt.isNotEmpty()) {
                        // Modify the API response to include context and mood
                        val context = when {
                            previousQuestions.size >= 2 -> {
                                val lastTwoTopics = previousQuestions.takeLast(2)
                                when (userMood) {
                                    "positive" -> "I'm excited to continue our discussion about ${lastTwoTopics[0]} and ${lastTwoTopics[1]}. $engagePrompt"
                                    "negative" -> "I understand this might be complex. Let's explore how ${lastTwoTopics[0]} and ${lastTwoTopics[1]} connect. $engagePrompt"
                                    else -> "Building on your questions about ${lastTwoTopics[0]} and ${lastTwoTopics[1]}, $engagePrompt"
                                }
                            }
                            else -> engagePrompt
                        }
                        context
                    } else {
                        // Generate contextual fallback based on conversation history and mood
                        val recentTopics = previousQuestions.takeLast(2)
                        when (userMood) {
                            "positive" -> "I'm really enjoying our discussion about ${recentTopics.joinToString(" and ")}. What would you like to explore next?"
                            "negative" -> "I understand this might be challenging. Let's explore ${recentTopics.joinToString(" and ")} further. What would you like to know?"
                            else -> "I notice you're particularly interested in ${recentTopics.joinToString(" and ")}. What would you like to explore next?"
                        }
                    }
                } catch (e: Exception) {
                    // Fallback with context from previous questions and mood
                    val recentTopics = previousQuestions.takeLast(2)
                    when (userMood) {
                        "positive" -> "I'm really enjoying our discussion about ${recentTopics.joinToString(" and ")}. What would you like to explore next?"
                        "negative" -> "I understand this might be challenging. Let's explore ${recentTopics.joinToString(" and ")} further. What would you like to know?"
                        else -> "I notice you're particularly interested in ${recentTopics.joinToString(" and ")}. What would you like to explore next?"
                    }
                }
            }
        }
        
        // Ask the follow-up question with appropriate gesture
        when (userMood) {
            "positive" -> furhat.gesture(Gestures.Smile)
            "negative" -> furhat.gesture(Gestures.ExpressSad)
            else -> furhat.gesture(Gestures.Nod)
        }
        furhat.ask(followUpPrompt)
    }

    onNoResponse {
        furhat.gesture(Gestures.ExpressSad)
        furhat.say("I didn't catch that. Could you please repeat your question?")
        reentry()
    }
}

// Helper function to call the /ask endpoint.
private fun callDocumentAgent(question: String): String {
    val baseUrl = LOCAL_BACKEND_URL
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
    val baseUrl = LOCAL_BACKEND_URL
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