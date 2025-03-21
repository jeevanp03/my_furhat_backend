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

    onEntry {
        // Lock the attended user during this conversation.
        furhat.say("Hello! What would you like to know about $documentName?")
    }

    onExit {
        // Release the attention lock when leaving this state.
    }

    onResponse<Goodbye> {
        furhat.say("Goodbye!")
    }

    onResponse {
        val userQuestion = it.text.trim()
        // Signal the robot is thinking with a filler utterance and gesture.
        furhat.say(async = true) {
            +"Let me think..."
            +Gestures.GazeAway(duration = 1.0)
        }
        // Call the backend /ask endpoint to get an answer.
        val answer = callDocumentAgent(userQuestion)
        furhat.say(answer)

        // Now call the new "engage user" API to get a dynamic follow-up prompt.
        val followUpPrompt = callEngageUser(documentName, answer)
        furhat.ask(followUpPrompt)
    }

    onResponse<Yes> {
        reentry()
    }

    onResponse<No> {
        furhat.say("Alright, thank you for the conversation. Goodbye!")
    }

    onNoResponse {
        furhat.ask("I didn't catch that. Could you please repeat your question?")
        reentry()
    }
}

// Helper function to call the /ask endpoint.
fun callDocumentAgent(question: String): String {
    val baseUrl = "http://localhost:8000"
    val client = OkHttpClient.Builder()
        .connectTimeout(60, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
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
// This function sends the current document context to the /engage endpoint
// and expects a JSON response with a "prompt" field containing a follow-up question.
fun callEngageUser(documentName: String, answer: String): String {
    val baseUrl = "http://localhost:8000"
    val client = OkHttpClient()
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
            jsonObject.getString("prompt")
        }
    } catch (e: ConnectException) {
        println(e)
        "Do you have any other questions?"
    }
}