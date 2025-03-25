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
import furhatos.app.templateadvancedskill.params.AWS_SERVER_URL
import java.util.concurrent.TimeUnit

// Document Q&A state, inheriting from Parent.
fun documentInfoQnA(documentName: String): State = state(parent = Parent) {
    var conversationCount = 0
    var lastQuestion = ""
    var lastAnswer = ""
    var previousQuestions = mutableListOf<String>()
    var previousAnswers = mutableListOf<String>()

    onEntry {
        // Lock the attended user during this conversation.
        furhat.say("Hello! I'm here to help you learn about $documentName. What would you like to know?")
    }

    onExit {
        // Release the attention lock when leaving this state.
    }

    onResponse<Goodbye> {
        furhat.say("Thank you for the interesting conversation! Goodbye!")
        goto(Idle)
    }

    onResponse<No> {
        furhat.say("Alright, thank you for the conversation. Goodbye!")
        goto(Idle)
    }

    onResponse {
        val userQuestion = it.text.trim()
        conversationCount++
        
        // Signal the robot is thinking
        furhat.say("Let me think about that...")
        furhat.gesture(Gestures.GazeAway(duration = 3.0))
        
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
        
        // Speak the cleaned answer
        furhat.say(cleanAnswer)
        
        // Generate a contextual follow-up based on conversation history
        val followUpPrompt = when {
            conversationCount == 1 -> {
                // First follow-up: Focus on specific aspects mentioned in the answer
                val keyAspects = cleanAnswer.split(".").take(2).joinToString(" ")
                "I notice you're interested in $keyAspects. What would you like to know more about that?"
            }
            conversationCount == 2 -> {
                // Second follow-up: Connect to previous question
                val previousTopic = previousQuestions[0].split(" ").take(3).joinToString(" ")
                "You asked about $previousTopic earlier. Would you like to explore how that relates to what we just discussed?"
            }
            else -> {
                // Later follow-ups: Use conversation history for context
                try {
                    // Try to get an engaging follow-up from the API
                    val engagePrompt = callEngageUser(documentName, cleanAnswer)
                    if (engagePrompt.isNotEmpty()) {
                        // Modify the API response to include context
                        val context = when {
                            previousQuestions.size >= 2 -> {
                                val lastTwoTopics = previousQuestions.takeLast(2)
                                "Building on your questions about ${lastTwoTopics[0]} and ${lastTwoTopics[1]}, $engagePrompt"
                            }
                            else -> engagePrompt
                        }
                        context
                    } else {
                        // Generate contextual fallback based on conversation history
                        val recentTopics = previousQuestions.takeLast(2)
                        "I notice you're particularly interested in ${recentTopics.joinToString(" and ")}. What would you like to explore next?"
                    }
                } catch (e: Exception) {
                    println("Error in engage API call: ${e.message}")
                    // Fallback with context from previous questions
                    val recentTopics = previousQuestions.takeLast(2)
                    "I notice you're particularly interested in ${recentTopics.joinToString(" and ")}. What would you like to explore next?"
                }
            }
        }
        
        // Ask the follow-up question
        furhat.ask(followUpPrompt)
    }

    onNoResponse {
        furhat.say("I didn't catch that. Could you please repeat your question?")
        reentry()
    }
}

// Helper function to call the /ask endpoint.
fun callDocumentAgent(question: String): String {
    val baseUrl = "http://localhost:8000"
    val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()
    return try {
        val requestBody = JSONObject().put("content", question).toString().toRequestBody("application/json; charset=utf-8".toMediaType())
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
        println(e)
        "I'm sorry, I cannot process your request right now."
    }
}

// New helper function to call the "engage user" API endpoint.
fun callEngageUser(documentName: String, answer: String): String {
    val baseUrl = "http://localhost:8000"
    val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()
    return try {
        val map = JSONObject()
        map.put("document", documentName)
        map.put("answer", answer)
        val requestBody = map.toString().toRequestBody("application/json; charset=utf-8".toMediaType())

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
                println("Error parsing prompt from response: $jsonResponse")
                ""  // Return empty string to trigger fallback question
            }
        }
    } catch (e: ConnectException) {
        println(e)
        ""  // Return empty string to trigger fallback question
    }
}