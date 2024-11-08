---
layout: default
permalink: /secret
---

<style>
        /* Basic styling for the form */
        .pin-container {
            display: flex;
            justify-content: center;
            gap: 10px;
        }
        .pin-input {
            width: 30px;
            font-size: 20px;
            text-align: center;
        }
        .image-container {
            display: none;
            margin-top: 20px;
            text-align: center;
        }
        #success-image {
            max-width: 100%;
            height: auto;
        }
    </style>


It's guessing time!


<form id="pin-form">
  <div class="pin-container">
   <input type="text" maxlength="2" class="pin-input" id="digit1" required>
   <input type="text" maxlength="2" class="pin-input" id="digit2" required>
   <input type="text" maxlength="2" class="pin-input" id="digit3" required>
   <input type="text" maxlength="3" class="pin-input" id="digit4" required>
  </div>
  <button type="button" onclick="verifyPin()">Submit</button>
</form>

<div class="image-container" id="image-container">
   <p>Correct PIN!</p>
   <img src="{{ site.url }}/_data/photo_clue.jpg" id="success-image" alt="Success Image">
</div>

  <script>
        function verifyPin() {
            const correctPin = "244481149"; // Set the correct PIN here
            const enteredPin = 
                document.getElementById("digit1").value + 
                document.getElementById("digit2").value + 
                document.getElementById("digit3").value + 
                document.getElementById("digit4").value;

            if (enteredPin === correctPin) {
                document.getElementById("image-container").style.display = "block";
            } else {
                alert("Incorrect PIN. Please try again.");
            }
        }
  </script>


