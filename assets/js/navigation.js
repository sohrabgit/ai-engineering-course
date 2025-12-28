// AI Engineering Course - Shared JavaScript

// Smooth scrolling for navigation links
document.addEventListener('DOMContentLoaded', function() {
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Navigation bar background on scroll
    const navbar = document.getElementById('navbar');
    if (navbar) {
        window.addEventListener('scroll', () => {
            if (window.scrollY > 100) {
                navbar.style.background = 'rgba(10, 10, 15, 0.98)';
            } else {
                navbar.style.background = 'rgba(10, 10, 15, 0.95)';
            }
        });
    }

    // Intersection Observer for animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, observerOptions);

    // Observe elements for animations
    document.querySelectorAll('.card, .content-section, .highlight-box').forEach(el => {
        observer.observe(el);
    });

    // Back to top functionality
    createBackToTopButton();

    // Mobile menu toggle (if needed)
    initializeMobileMenu();

    // Progress tracking
    initializeProgressTracking();
});

// Back to top button
function createBackToTopButton() {
    let backToTopButton = document.createElement('button');
    backToTopButton.innerHTML = '↑';
    backToTopButton.className = 'back-to-top';
    backToTopButton.style.cssText = `
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        width: 50px;
        height: 50px;
        border: none;
        border-radius: 50%;
        background: var(--gradient-primary);
        color: white;
        font-size: 1.5rem;
        cursor: pointer;
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s ease;
        z-index: 1000;
        box-shadow: var(--shadow-glow);
    `;
    
    backToTopButton.addEventListener('click', () => {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });

    document.body.appendChild(backToTopButton);

    window.addEventListener('scroll', () => {
        if (window.scrollY > 500) {
            backToTopButton.style.opacity = '1';
            backToTopButton.style.visibility = 'visible';
        } else {
            backToTopButton.style.opacity = '0';
            backToTopButton.style.visibility = 'hidden';
        }
    });
}

// Mobile menu initialization
function initializeMobileMenu() {
    const mobileMenuButton = document.createElement('button');
    mobileMenuButton.innerHTML = '☰';
    mobileMenuButton.className = 'mobile-menu-toggle';
    mobileMenuButton.style.cssText = `
        display: none;
        background: none;
        border: none;
        color: var(--text-primary);
        font-size: 1.5rem;
        cursor: pointer;
        padding: 0.5rem;
    `;

    // Add mobile menu styles
    const mobileMenuStyles = `
        @media (max-width: 768px) {
            .mobile-menu-toggle {
                display: block !important;
            }
            .nav-links.mobile-open {
                display: flex !important;
                flex-direction: column;
                position: absolute;
                top: 100%;
                left: 0;
                right: 0;
                background: var(--background);
                border-top: 1px solid var(--border-color);
                padding: 1rem;
                gap: 1rem;
            }
        }
    `;

    const styleSheet = document.createElement('style');
    styleSheet.textContent = mobileMenuStyles;
    document.head.appendChild(styleSheet);

    const navLinks = document.querySelector('.nav-links');
    if (navLinks) {
        const navContent = document.querySelector('.nav-content');
        navContent.appendChild(mobileMenuButton);

        mobileMenuButton.addEventListener('click', () => {
            navLinks.classList.toggle('mobile-open');
        });
    }
}

// Progress tracking functionality
function initializeProgressTracking() {
    // Track reading progress on content pages
    if (document.querySelector('.content-page')) {
        let progressBar = document.createElement('div');
        progressBar.className = 'reading-progress';
        progressBar.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 0%;
            height: 3px;
            background: var(--gradient-primary);
            z-index: 1001;
            transition: width 0.1s ease;
        `;
        document.body.appendChild(progressBar);

        window.addEventListener('scroll', () => {
            const scrolled = (window.scrollY / (document.documentElement.scrollHeight - window.innerHeight)) * 100;
            progressBar.style.width = Math.min(scrolled, 100) + '%';
        });
    }
}

// Course navigation helpers
function navigateToWeek(weekNumber) {
    window.location.href = `./weeks/week${weekNumber}/index.html`;
}

function navigateToSession(weekNumber, sessionNumber) {
    window.location.href = `./weeks/week${weekNumber}/session${sessionNumber}/index.html`;
}

function navigateToTopic(weekNumber, sessionNumber, topicNumber) {
    window.location.href = `./weeks/week${weekNumber}/session${sessionNumber}/topic${topicNumber}.html`;
}

// Accordion functionality for expandable content
function initializeAccordions() {
    document.querySelectorAll('.accordion-header').forEach(header => {
        header.addEventListener('click', function() {
            const accordionItem = this.parentElement;
            const content = accordionItem.querySelector('.accordion-content');
            const isActive = accordionItem.classList.contains('active');
            
            // Close all other accordions in the same group
            const group = accordionItem.closest('.accordion-group');
            if (group) {
                group.querySelectorAll('.accordion-item').forEach(item => {
                    item.classList.remove('active');
                    const itemContent = item.querySelector('.accordion-content');
                    if (itemContent) {
                        itemContent.style.maxHeight = null;
                    }
                });
            }
            
            // Toggle current accordion
            if (!isActive) {
                accordionItem.classList.add('active');
                if (content) {
                    content.style.maxHeight = content.scrollHeight + 'px';
                }
            }
        });
    });
}

// Course completion tracking (using localStorage)
class CourseProgress {
    constructor() {
        this.storageKey = 'ai-course-progress';
        this.progress = this.loadProgress();
    }

    loadProgress() {
        const stored = localStorage.getItem(this.storageKey);
        return stored ? JSON.parse(stored) : {};
    }

    saveProgress() {
        localStorage.setItem(this.storageKey, JSON.stringify(this.progress));
    }

    markTopicComplete(weekNumber, sessionNumber, topicNumber) {
        const key = `week${weekNumber}-session${sessionNumber}-topic${topicNumber}`;
        this.progress[key] = {
            completed: true,
            completedAt: new Date().toISOString()
        };
        this.saveProgress();
        this.updateProgressDisplay();
    }

    isTopicComplete(weekNumber, sessionNumber, topicNumber) {
        const key = `week${weekNumber}-session${sessionNumber}-topic${topicNumber}`;
        return this.progress[key] && this.progress[key].completed;
    }

    updateProgressDisplay() {
        // Update progress indicators on the page
        document.querySelectorAll('[data-progress]').forEach(element => {
            const [week, session, topic] = element.dataset.progress.split('-');
            if (this.isTopicComplete(week, session, topic)) {
                element.classList.add('completed');
            }
        });
    }

    getOverallProgress() {
        const totalTopics = 12; // Adjust based on actual course structure
        const completedTopics = Object.keys(this.progress).length;
        return Math.round((completedTopics / totalTopics) * 100);
    }
}

// Initialize course progress tracking
let courseProgress;
document.addEventListener('DOMContentLoaded', () => {
    courseProgress = new CourseProgress();
    courseProgress.updateProgressDisplay();
});

// Export functions for global use
window.CourseNavigation = {
    navigateToWeek,
    navigateToSession,
    navigateToTopic,
    courseProgress: () => courseProgress
};