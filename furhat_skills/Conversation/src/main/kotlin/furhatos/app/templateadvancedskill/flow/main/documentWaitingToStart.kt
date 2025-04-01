package furhatos.app.templateadvancedskill.flow.main

import furhatos.flow.kotlin.*
import furhatos.app.templateadvancedskill.flow.Parent
import furhatos.app.templateadvancedskill.params
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import org.json.JSONObject
import java.io.IOException
import okhttp3.*
import furhatos.app.templateadvancedskill.params.LOCAL_BACKEND_URL
import furhatos.app.templateadvancedskill.params.AWS_BACKEND_URL
import java.net.ConnectException
import java.net.SocketTimeoutException
import java.util.concurrent.TimeUnit


val DocumentWaitingToStart: State = state(parent = Parent) {
    onEntry {
        furhat.ask("I'm ready to assist with your document questions. Could you please tell me what subject you're interested in, or simply the name of the document?")
    }

    // When any response is detected, transition to SelectDocument.
    onResponse {
//        goto(SelectDocument)
        val userInput = it.text.trim()
        // Call the API endpoint /get_docs to perform document retrieval/classification.
        val bestDocName = callGetDocs(userInput)
        // Transition to the Q&A state for the matched document.
        goto(documentInfoQnA(bestDocName))
    }

    onNoResponse {
        furhat.ask("I didn't catch that. Please tell me the subject or the name of the document you're interested in.")
        reentry()
    }
}

fun callGetDocs(userInput: String): String {
    // Your FastAPI server's address (adjust if needed)
    val url = "$AWS_BACKEND_URL/get_docs"
    val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    // Build JSON payload.
    val jsonBody = """{"content":"$userInput"}"""
    val body = RequestBody.create("application/json".toMediaType(), jsonBody)
    val request = Request.Builder()
        .url(url)
        .post(body)
        .build()

    return try {
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw IOException("Unexpected response: $response")
            }
            val respString = response.body?.string() ?: throw IOException("Empty response body")
            val json = JSONObject(respString)
            json.getString("response")
        }
    } catch (e: ConnectException) {
        "I'm sorry, I cannot connect to the server right now. Please try again later."
    } catch (e: SocketTimeoutException) {
        "I'm sorry, the server is taking too long to respond. Please try again later."
    } catch (e: Exception) {
        "I apologize, but I encountered an error processing your request."
    }
}

