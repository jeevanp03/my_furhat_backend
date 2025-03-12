package furhatos.app.templateadvancedskill.flow.main

import furhatos.flow.kotlin.*
// Import the documentInfoQnA state from its package (adjust the package path if needed)
import furhatos.app.documentagent.flow.documentInfoQnA
import furhatos.app.templateadvancedskill.flow.Parent
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import org.json.JSONObject
import java.io.IOException

val SelectDocument: State = state(parent = Parent) {
    onEntry {
        // A more natural prompt asking the user for either subject matter or document name.
        furhat.ask("Could you please tell me what subject you're interested in, or simply the name of the document?")
    }

    // Capture any user response and use the /get_docs API to find the best match.
    onResponse {
        val userInput = it.text.trim()
        // Call the API endpoint /get_docs to perform document retrieval/classification.
        val bestDocName = callGetDocs(userInput)
        // Transition to the Q&A state for the matched document.
        goto(documentInfoQnA(bestDocName))
    }

    onNoResponse {
        furhat.ask("I didn't catch that. Please tell me the subject or name of the document you're interested in.")
        reentry()
    }
}

// Helper function to call the /get_docs endpoint.
// This function sends a POST request with the user's input as JSON to /get_docs,
// then parses and returns the 'response' field from the returned JSON.
fun callGetDocs(userInput: String): String {
    // Your FastAPI server's address (adjust if needed)
    val url = "http://localhost:8000"
    val client = OkHttpClient()

    // Build JSON payload.
    val jsonBody = """{"content":"$userInput"}"""
    val body = RequestBody.create("application/json".toMediaType(), jsonBody)
    val request = Request.Builder()
            .url(url)
            .post(body)
            .build()

    client.newCall(request).execute().use { response ->
        if (!response.isSuccessful) {
            throw IOException("Unexpected response: $response")
        }
        val respString = response.body?.string() ?: throw IOException("Empty response body")
        val json = JSONObject(respString)
        return json.getString("response")
    }
}
