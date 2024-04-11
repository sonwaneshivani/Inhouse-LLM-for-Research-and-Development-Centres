function processText(action) {
    var textInput = document.getElementById("customTextInput").value;
    var fileInput = document.getElementById("customFileInput").files[0];
    var questionsInput = document.getElementById("customQuestionsInput").value;
  
    document.getElementById("custom-loadingIcon").style.display = "inline-block";
  
    var formData = new FormData();
    formData.append("text", textInput);
    formData.append("file", fileInput);
    formData.append("action", action);
  
    if (action === "generateQuestions") {
      formData.append("questionsInput", questionsInput);
    }
  
    $.ajax({
      url: "/process_text",
      type: "POST",
      data: formData,
      contentType: false,
      processData: false,
      success: function (data) {
        document.getElementById("custom-loadingIcon").style.display = "none";
        document.getElementById("custom-outputText").innerHTML = data;
      },
      error: function (error) {
        console.error("Error:", error);
      },
    });
  }
  
  function toggleQuestionsInput() {
    var questionsContainer = document.getElementById("customQuestionsContainer");
    questionsContainer.style.display =
      questionsContainer.style.display === "none" ? "block" : "none";
  }
  
  function generateQuestions() {
    processText("generateQuestions");
  }
  
  
  function toggleDarkMode() {
    var body = document.body;
    body.classList.toggle("dark-mode");
  
    var navbar = document.querySelector(".custom-navbar");
    navbar.classList.toggle("dark-mode-navbar");
  
    var elementsToToggle = document.querySelectorAll(".custom-card, .custom-form-control, .custom-label, #custom-outputContainer");
    elementsToToggle.forEach(element => {
      element.classList.toggle("dark-mode-elements");
    });
  
    // Add or remove dark mode to the body margin
    body.classList.toggle("dark-mode-body-margin");
}
  