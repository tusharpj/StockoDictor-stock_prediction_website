// Initialize Firebase
const firebaseConfig = {
    apiKey: "YOUR_API_KEY",  // Replace with your Firebase API key
    authDomain: "YOUR_PROJECT_ID.firebaseapp.com",
    projectId: "YOUR_PROJECT_ID",
    storageBucket: "YOUR_PROJECT_ID.appspot.com",
    messagingSenderId: "YOUR_SENDER_ID",
    appId: "YOUR_APP_ID",
  };
  
  const app = firebase.initializeApp(firebaseConfig);
  const auth = firebase.auth();
  const storage = firebase.storage();
  
  // Handle registration with animation
  document.querySelector("form").addEventListener("submit", function (e) {
    e.preventDefault();
  
    const name = document.querySelector("#name").value;
    const email = document.querySelector("#email").value;
    const password = document.querySelector("#password").value;
  
    // Form input validation (e.g., simple email validation)
    if (email === "" || password === "" || name === "") {
      alert("Please fill in all fields.");
      return;
    }
  
    // Show a loading animation (You can replace this with a custom loader)
    document.querySelector("button").innerHTML = "Registering...";
    document.querySelector("button").disabled = true;
  
    // Create user with email and password
    auth
      .createUserWithEmailAndPassword(email, password)
      .then((userCredential) => {
        // Registration successful, create user profile
        const user = userCredential.user;
        const displayName = name;
        
        // Update user's display name
        user.updateProfile({
          displayName: displayName,
        }).then(() => {
          // Registration successful
          alert("Registration successful!");
          // Redirect to login page or homepage
          window.location.href = "/pages/login.html";
        });
      })
      .catch((error) => {
        const errorMessage = error.message;
        alert("Error: " + errorMessage);
      })
      .finally(() => {
        document.querySelector("button").innerHTML = "Register";
        document.querySelector("button").disabled = false;
      });
  });
  
  // Add some animation effects using GSAP or plain JavaScript
  document.addEventListener("DOMContentLoaded", () => {
    // Navbar animation
    const navbar = document.querySelector(".navbar");
    navbar.style.opacity = 0;
    setTimeout(() => {
      navbar.style.transition = "opacity 1s ease-in-out";
      navbar.style.opacity = 1;
    }, 500);
  
    // Form fade-in animation
    const formContainer = document.querySelector(".form-container");
    formContainer.style.opacity = 0;
    formContainer.style.transform = "translateY(50px)";
    setTimeout(() => {
      formContainer.style.transition = "all 1s ease-out";
      formContainer.style.opacity = 1;
      formContainer.style.transform = "translateY(0)";
    }, 1000);
  });
  