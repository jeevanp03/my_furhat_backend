package furhatos.app.documentagent.flow

import furhatos.flow.kotlin.*
import okhttp3.*
import org.json.JSONObject
import java.net.ConnectException
import okio.IOException
import furhatos.event.Event
import furhatos.flow.kotlin.*
import furhatos.nlu.common.*
import furhatos.app.templateadvancedskill.flow.Parent
import furhatos.gestures.Gestures
import kotlinx.coroutines.delay

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

//    onUserSilence {
//        furhat.say("I'm still here—please ask your question when you're ready.")
//        reentry()
//    }
}

// Helper function to call the /ask endpoint.
fun callDocumentAgent(question: String): String {
    val baseUrl = "http://localhost:8000/ask"
    val client = OkHttpClient()
    return try {
        val request = Request.Builder()
                .url("$baseUrl/ask")
                .post(FormBody.Builder().add("content", question).build())
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
