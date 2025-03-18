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

        furhat.ask("Hello! What would you like to know about $documentName?")
    }

    onExit {
        // Release the attention lock when leaving this state.

    }

//    onResponse<Stop> {
//        furhat.say("Okay, ending the conversation. Goodbye!")
//
//    }

    onResponse<Goodbye> {
        furhat.say("Goodbye!")

    }

    onResponse<Yes> {
        reentry()
    }

    onResponse<No> {
        furhat.say("Alright, thank you for the conversation. Goodbye!")

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
        furhat.ask("Do you have another question about $documentName?")
    }

    onNoResponse {
        furhat.ask("I didn't catch that. Could you please repeat your question?")
        reentry()
    }

//    onUserSilence {
//        furhat.say("I'm still here—please ask your question when you're ready.")
//        reentry()
//    }
}

// Helper function to call the /ask endpoint.
fun callDocumentAgent(question: String): String {
    val baseUrl = "http://$LOCAL_BACKEND_URL:8000"
    // Configure OkHttpClient with longer timeouts
    val client = OkHttpClient.Builder()
        .connectTimeout(60, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()

    return try {
        // Create JSON payload
        val jsonPayload = JSONObject().put("content", question).toString()
        val requestBody = jsonPayload.toRequestBody("application/json; charset=utf-8".toMediaType())

        val request = Request.Builder()
            .url("$baseUrl/ask")
            .post(requestBody)
            .build()

        val response = client.newCall(request).execute()
        if (!response.isSuccessful) {
            throw IOException("Unexpected response: $response")
        }
        val jsonResponse = response.body?.string() ?: throw IOException("Empty response")
        val jsonObject = JSONObject(jsonResponse)
        jsonObject.getString("response")
    } catch (e: ConnectException) {
        println(e)
        "I'm sorry, I cannot process your request right now."
    }
}