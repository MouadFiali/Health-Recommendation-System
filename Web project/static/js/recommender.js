self.addEventListener('message', (event) => {
    console.log("received: event", event);
    const getAnswer = (prompt) => {
        event.preventDefault();
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
            let answer = data.response;
             // replace '\n' with '<br>' to display in html
            answer = answer.replace(/\n/g, '<br>');
            self.postMessage(answer);
        })
        .catch(error => {
            
        });
    }  
    const { data } = event;
    console.log("received: data", data);

    getAnswer(data);
});