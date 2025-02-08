document.addEventListener("DOMContentLoaded", function() {

    // GSAP animation for the navbar
    gsap.from(".navbar", {
      duration: 1,
      opacity: 0,
      y: -50, // Start slightly above the viewport and slide in
      ease: "power2.out"
    });
  
    // GSAP animation for the site name (logo section)
    gsap.from(".site-name", {
      duration: 1,
      opacity: 0,
      x: -100, // Start from left
      ease: "power2.out",
      delay: 0.5
    });
  
    // GSAP animation for navigation buttons
    gsap.from(".nav-links .btn", {
      duration: 0.8,
      opacity: 0,
      scale: 0.8,
      stagger: 0.3, // Stagger the animations for each button
      ease: "power2.out",
      delay: 1
    });
  
    // Animation for the header when it loads
    gsap.from("header h1", {
      duration: 1.2,
      opacity: 0,
      y: 50, // Start below and move up
      ease: "power2.out",
      delay: 1.5
    });
  
    // Add hover effect for buttons
    const buttons = document.querySelectorAll(".nav-links .btn");
  
    buttons.forEach(button => {
      button.addEventListener("mouseenter", () => {
        gsap.to(button, {
          scale: 1.1,
          backgroundColor: "#0056b3",
          duration: 0.3,
          ease: "power1.out"
        });
      });
  
      button.addEventListener("mouseleave", () => {
        gsap.to(button, {
          scale: 1,
          backgroundColor: "#007bff",
          duration: 0.3,
          ease: "power1.out"
        });
      });
    });
  
  });
  