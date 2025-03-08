document.addEventListener("DOMContentLoaded", function() {
    // Add links to navigation boxes
    document.querySelectorAll(".toms-nav-container .box, .toms-nav-box").forEach(element => {
        element.addEventListener("click", function() {
            window.location.href = this.getAttribute("data-href");
        });
    });

    // Hide dummy home/feedback titles if they exist
    ["home", "feedback"].forEach(dummy => {
        let dummyElement = document.getElementById(dummy);
        if (dummyElement) {
            let title = dummyElement.querySelector("h1");
            if (title) {
                title.style.display = "none";
            }
        }
    });

    // Adjust tutorial section title if active
    let activeEl = document.querySelector(".bd-navbar-elements .navbar-nav .nav-item.active");
    if (activeEl && activeEl.innerText.trim() === "Tutorials") {
        let tutorialTitle = document.querySelector(".bd-links__title");
        if (tutorialTitle) {
            tutorialTitle.innerText = "Other tutorials";
        }
    }

    // Add tqdm classes for stderr messages containing progress indicators
    document.querySelectorAll(".stderr").forEach(x => {
        if (x.innerText.includes("it/s")) {
            x.classList.add("tqdm");
        }
    });

    // Adjust navbar header layout for better responsiveness
    let start = document.querySelector(".navbar-header-items__start");
    if (start) {
        start.classList.remove("col-lg-3");
        start.classList.add("col-lg");
    }
    
    let middle = document.querySelector(".navbar-header-items");
    if (middle) {
        middle.classList.remove("col-lg-9");
        middle.classList.add("col-lg-10");
    }

    // Ensure proper visibility for light/dark mode images
    document.querySelectorAll("figure").forEach(el => {
        if (el.querySelector(".only-light")) {
            el.classList.add("only-light");
        }
        if (el.querySelector(".only-dark")) {
            el.classList.add("only-dark");
        }
    });

    // Collapse section navigation dropdown when in Tutorials
    let nav = document.querySelector('.bd-docs-nav[aria-label="Section Navigation"]');
    if (nav) {
        let firstNavItem = nav.querySelector("a");
        if (firstNavItem && firstNavItem.innerText.trim() === "Tutorials") {
            let label = nav.querySelector("label");
            if (label) {
                label.click();
            }
        }
    }

    // Remove h3 elements in User Guide for cleaner layout
    let userGuideActive = document.querySelector(".navbar-nav .active");
    if (userGuideActive && userGuideActive.innerText.trim() === "User Guide") {
        document.querySelectorAll("h3").forEach(el => el.remove());
    }
});
