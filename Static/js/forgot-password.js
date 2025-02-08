// forgot-password.js

// Firebase configuration
const firebaseConfig = {
    apiKey: "your-api-key",
    authDomain: "your-auth-domain",
    projectId: "your-project-id",
    storageBucket: "your-storage-bucket",
    messagingSenderId: "your-sender-id",
    appId: "your-app-id"
  };
  
  // Initialize Firebase
  firebase.initializeApp(firebaseConfig);
  
  // Get elements from the DOM
  const resetPasswordForm = document.getElementById('reset-password-form');
  const emailInput = document.getElementById('email');
  
  // Handle form submission
  resetPasswordForm.addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent form submission
  
    const email = emailInput.value;
  
    if (email) {
      // Send password reset email using Firebase
      firebase.auth().sendPasswordResetEmail(email)
        .then(function() {
          alert('Password reset link sent! Please check your email.');
          resetPasswordForm.reset(); // Reset the form
        })
        .catch(function(error) {
          const errorCode = error.code;
          const errorMessage = error.message;
          alert('Error: ' + errorMessage);
        });
    } else {
      alert('Please enter a valid email.');
    }
  });
  