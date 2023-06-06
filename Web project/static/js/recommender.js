self.addEventListener('message', (event) => {
    console.log("received: event", event);
    const getAnswer = (prompt) => {
        event.preventDefault();
        var response = "reponse";
        var dateTime = new Date();
        // var time = dateTime.toLocaleTimeString();
        // Add the prompt to the response div
        // $('#response').append('<p id="GFG1">('+ time + ') <i class="bi bi-person"></i>: ' + prompt + '</p>');
        // $('#response #GFG1').css({"color": "green", "width": "90%", "float": "left"});
    
        fetch('/predict', {
            method: 'POST',
            body: JSON.stringify({ prompt: prompt }),
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            // $('#response').append('<p id="GFG2">('+ time + ') <i class="bi bi-robot"></i>: ' + data.response + '</p>');
            // $('#response #GFG2').css({"color": "red", "width": "90%", "float": "right"});
            console.log(data.response);
            response = data.response;
        })
        .catch(error => {
            
        });
        return response;
    }  
    const { data } = event;
    console.log("received: data", data);

    const answer = getAnswer(data);
    self.postMessage(answer);
});