// Firebase Authentication and Storage
import { getAuth, signInWithEmailAndPassword } from "https://www.gstatic.com/firebasejs/9.1.1/firebase-auth.js";
import { getStorage, ref, uploadBytes } from "https://www.gstatic.com/firebasejs/9.1.1/firebase-storage.js";

// Get references to the form elements
const loginForm = document.querySelector('form');
const emailInput = document.getElementById('email');
const passwordInput = document.getElementById('password');
const forgotPasswordLink = document.querySelector('.forgot-password a');
const registerLink = document.querySelector('.no-account a');

// Get the Firebase Auth and Storage instances
const auth = getAuth();
const storage = getStorage();

// Handle form submission
loginForm.addEventListener('submit', async (event) => {
  event.preventDefault(); // Prevent form from submitting the traditional way

  const email = emailInput.value;
  const password = passwordInput.value;

  try {
    // Sign in with Firebase Authentication
    const userCredential = await signInWithEmailAndPassword(auth, email, password);
    const user = userCredential.user;
    console.log('User logged in:', user);

    // Redirect to home page on successful login
    window.location.href = '/pages/home.html'; // Replace with your home page URL
    window.location.href = 'home.html'; 

  } catch (error) {
    const errorCode = error.code;
    const errorMessage = error.message;
    alert(`Error: ${errorMessage}`); // Display error message
  }
});

// Forgot Password functionality
forgotPasswordLink.addEventListener('click', () => {
  const email = prompt('Please enter your email to reset the password:');
  if (email) {
    // Send password reset email
    firebase.auth().sendPasswordResetEmail(email)
      .then(() => {
        alert('Password reset email sent!');
      })
      .catch((error) => {
        console.error('Error sending password reset email: ', error);
        alert('Error: ' + error.message);
      });
  }
});

// Register functionality (Redirect to register page)
registerLink.addEventListener('click', () => {
  window.location.href = '/pages/register.html'; // Redirect to register page
});
