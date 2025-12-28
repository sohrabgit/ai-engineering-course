/**
 * Common JavaScript functionality for topic pages
 * This file reduces code duplication across topic pages
 */

// Reading progress tracking - common across all topic pages
function initializeReadingProgress() {
    const progressIndicator = document.getElementById('reading-progress');
    
    if (progressIndicator) {
        window.addEventListener('scroll', () => {
            const scrolled = (window.scrollY / (document.documentElement.scrollHeight - window.innerHeight)) * 100;
            progressIndicator.style.width = Math.min(scrolled, 100) + '%';
        });
    }
}

// Topic completion tracking
function initializeTopicCompletion(week, session, topic) {
    window.addEventListener('scroll', () => {
        const scrollPosition = window.scrollY + window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight;
        
        if (scrollPosition >= documentHeight * 0.9) {
            // Mark as completed in local storage
            if (window.CourseNavigation && window.CourseNavigation.courseProgress) {
                window.CourseNavigation.courseProgress().markTopicComplete(week, session, topic);
            }
        }
    });
}

// Smooth scrolling for internal links
function initializeSmoothScrolling() {
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
}

// Auto-initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeReadingProgress();
    initializeSmoothScrolling();
});

// Export functions for manual initialization if needed
window.TopicCommon = {
    initializeReadingProgress,
    initializeTopicCompletion,
    initializeSmoothScrolling
};