# AI Engineering Course Website

A comprehensive, hierarchical course website for a 6-week AI Engineering curriculum. Built as a static website optimized for GitHub Pages deployment with rich educational content and progressive course structure.

## 🎯 Project Overview

This project transforms traditional course materials into an interactive, navigable website with:
- **Hierarchical content organization** (Course → Weeks → Sessions → Topics)
- **Rich educational content** including lecture scripts, interactive elements, and instructor notes
- **Professional design** with Cyber Academic theme (dark mode with cyan/purple accents)
- **Mobile-responsive** design optimized for all devices
- **GitHub Pages ready** with zero external dependencies

## 📁 Project Structure

```
ai-engineering-course/
├── README.md                              # This documentation
├── index.html                             # Main course overview (navigation hub)
├── test-style.html                        # Style test page
├── run-local-server.py                    # Local development server
│
├── assets/                                # Shared resources
│   ├── css/
│   │   └── shared.css                     # Unified styling system
│   └── js/
│       └── navigation.js                  # Shared JavaScript functionality
│
├── weeks/                                 # Course content by week
│   └── week1/                             # Week 1: AI Fundamentals
│       ├── index.html                     # Week overview page
│       ├── project.html                   # Detailed project guide
│       └── session1-1/                    # Session 1.1 content
│           └── topic1.html                # "What is AI?" - full lesson content
│
└── course-content/                        # Source markdown files
    └── Week1_Session1.1_Topic1_What_is_AI.md  # Original content source
```

## 🚀 Quick Start

### Option 1: Local File Viewing
Open directly in browser:
```
file:///path/to/ai-engineering-course/index.html
```

### Option 2: Local Development Server (Recommended)
```bash
cd ai-engineering-course
python3 run-local-server.py
# Open http://localhost:8000
```

### Option 3: Simple Python Server
```bash
cd ai-engineering-course
python3 -m http.server 8000
# Open http://localhost:8000
```

### Option 4: GitHub Pages Deployment
1. Create GitHub repository
2. Upload all files maintaining directory structure
3. Enable Pages in Settings > Pages
4. Select 'main' branch and '/ (root)' folder
5. Site will be live at: `https://username.github.io/repo-name/`

## 🎨 Design System

### Color Scheme (Cyber Academic Theme)
```css
--primary-color: #00d4ff        /* Cyan blue */
--secondary-color: #7c3aed      /* Purple */
--accent-color: #10b981         /* Green */
--background: #0a0a0f           /* Dark background */
--surface: #1a1a2e              /* Card backgrounds */
--text-primary: #ffffff         /* Primary text */
--text-secondary: #94a3b8       /* Secondary text */
```

### Typography
- **Display Font**: Space Mono (monospace, tech-focused)
- **Body Font**: Inter Display (clean, readable)
- **Responsive**: Fluid typography with clamp()

### Components
- **Cards**: Rounded corners, hover effects, subtle shadows
- **Buttons**: Gradient backgrounds with glow effects
- **Navigation**: Sticky header with backdrop blur
- **Animations**: Smooth transitions, scroll-triggered reveals

## 📚 Content Structure & Implementation

### Current Implementation Status

#### ✅ Completed
- **Main Navigation Hub** (`index.html`)
  - Course overview with 6 week cards
  - Feature highlights and course description
  - Responsive grid layout

- **Week 1 Complete Structure**
  - **Overview Page** (`weeks/week1/index.html`)
    - Learning objectives
    - Session previews with duration and topics
    - Project overview with navigation
  
  - **Detailed Lesson Content** (`weeks/week1/session1-1/topic1.html`)
    - Full 20-minute lesson from markdown source
    - Instructor scripts and speaking notes
    - Interactive elements and discussion prompts
    - Student Q&A section
    - Visual aids and timeline graphics
    - Reading progress indicator
    - Assessment checkpoints
  
  - **Comprehensive Project Guide** (`weeks/week1/project.html`)
    - Detailed deliverables and timeline
    - Resources and evaluation criteria
    - Submission guidelines

#### 📋 Placeholder Structure (Weeks 2-6)
- Week overviews with realistic content
- Session structures ready for content addition
- Navigation pathways established

## 🔧 Adding New Content

### Adding a New Topic/Session

#### 1. Create Directory Structure
```bash
mkdir -p weeks/week[N]/session[N]-[N]
```

#### 2. Create Topic HTML File
Use the template structure from `weeks/week1/session1-1/topic1.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <!-- Standard header with relative paths -->
    <link rel="stylesheet" href="../../../assets/css/shared.css">
</head>
<body class="content-page">
    <!-- Navigation with breadcrumbs -->
    <!-- Content sections -->
    <!-- Footer -->
    <script src="../../../assets/js/navigation.js"></script>
</body>
</html>
```

#### 3. Essential Sections to Include
```html
<!-- Reading Progress Indicator -->
<div class="progress-indicator" id="reading-progress"></div>

<!-- Breadcrumb Navigation -->
<div class="breadcrumb">
    <div class="container">
        <a href="../../../">Home</a>
        <span class="breadcrumb-separator">→</span>
        <a href="../">Week [N]</a>
        <!-- ... -->
    </div>
</div>

<!-- Topic Hero with Meta Information -->
<section class="topic-hero">
    <div class="container">
        <div class="topic-meta">
            <span class="session-badge">Week [N], Session [N].[N], Topic [N]</span>
            <span class="duration-badge">[X] minutes</span>
        </div>
        <h1 class="topic-title section-title">[Topic Title]</h1>
    </div>
</section>

<!-- Learning Objectives -->
<section class="content-section">
    <div class="container">
        <h2>Learning Objectives</h2>
        <ul class="styled-list">
            <li>Objective 1</li>
            <li>Objective 2</li>
        </ul>
    </div>
</section>

<!-- Lecture Sections -->
<section class="content-section">
    <div class="container">
        <div class="lecture-section">
            <div class="lecture-header">
                <h3 class="lecture-title">Section Title</h3>
                <div class="lecture-timing">(X minutes)</div>
            </div>
            
            <!-- Content boxes for structured information -->
            <div class="content-box">
                <h4>Slide [N]: [Title]</h4>
                <code>[Slide content]</code>
            </div>
            
            <!-- Instructor scripts -->
            <div class="instructor-script">
                <p>[Speaking notes and explanations]</p>
            </div>
            
            <!-- Interactive elements -->
            <div class="interactive-element">
                <p><strong>[Activity title]:</strong> [Description]</p>
            </div>
        </div>
    </div>
</section>
```

### Content Component Types

#### 1. Lecture Sections
```html
<div class="lecture-section">
    <div class="lecture-header">
        <h3 class="lecture-title">Section Title</h3>
        <div class="lecture-timing">(Duration)</div>
    </div>
    <!-- Content -->
</div>
```

#### 2. Content Boxes (Slide Content)
```html
<div class="content-box">
    <h4>Slide Title</h4>
    <code>Structured content
Key points
• Bullet points</code>
</div>
```

#### 3. Instructor Scripts
```html
<div class="instructor-script">
    <p>"Speaking notes and detailed explanations that instructors should cover during this section."</p>
</div>
```

#### 4. Interactive Elements
```html
<div class="interactive-element">
    <p><strong>Activity:</strong> Description of hands-on activity or discussion prompt.</p>
</div>
```

#### 5. Visual Aids
```html
<div class="visual-aid">
    <h4>Visual Description</h4>
    <div class="timeline-visual">
        <!-- Custom visual content -->
    </div>
</div>
```

#### 6. Highlight Boxes
```html
<div class="highlight-box info">
    <h4>Important Note</h4>
    <p>Key information to emphasize.</p>
</div>

<div class="highlight-box warning">
    <h4>Warning</h4>
    <p>Important caution or consideration.</p>
</div>
```

#### 7. Q&A Sections
```html
<div class="qa-section">
    <h3 class="qa-title">Common Questions</h3>
    <div class="qa-item">
        <div class="qa-question">Q: Question text?</div>
        <div class="qa-answer">A: Answer text.</div>
    </div>
</div>
```

#### 8. Instructor Notes
```html
<div class="instructor-notes">
    <h4>Instructor Notes</h4>
    <p><strong>Timing:</strong> Important pacing information</p>
    <ul class="styled-list">
        <li>Teaching tip 1</li>
        <li>Teaching tip 2</li>
    </ul>
</div>
```

### Adding a New Week

#### 1. Create Week Directory
```bash
mkdir -p weeks/week[N]
mkdir -p weeks/week[N]/session[N]-1
mkdir -p weeks/week[N]/session[N]-2
```

#### 2. Create Week Overview Page
Copy and modify `weeks/week1/index.html`:

```html
<!-- Update all navigation paths -->
<link rel="stylesheet" href="../../assets/css/shared.css">

<!-- Update breadcrumb -->
<span class="breadcrumb-current">Week [N]</span>

<!-- Update week header -->
<div class="week-number">Week [N]</div>
<h1 class="week-title section-title">[Week Title]</h1>

<!-- Update session cards with appropriate hrefs -->
<a href="./session[N]-1/topic1.html" class="session-card">
```

#### 3. Update Main Navigation
Add week card to main `index.html`:

```html
<a href="./weeks/week[N]/" class="week-card">
    <div class="week-number">Week [N]</div>
    <h3 class="week-title">[Week Title]</h3>
    <p class="week-description">[Week Description]</p>
    <div class="week-stats">
        <span class="sessions-count">[X] Sessions</span>
        <span class="duration">[X] min total</span>
    </div>
</a>
```

## 🎯 Content Conversion Guidelines

### From Markdown to HTML

When converting existing markdown content (like the provided course files):

#### 1. Extract Structure
- **Week title** → Page title and hero section
- **Session headers** → Lecture sections
- **Learning objectives** → Styled lists
- **Duration** → Duration badges
- **Content blocks** → Content boxes or instructor scripts

#### 2. Identify Content Types
- **Lecture scripts** → `instructor-script` class
- **Slide content** → `content-box` class  
- **Activities** → `interactive-element` class
- **Q&A** → `qa-section` class
- **Notes** → `instructor-notes` class

#### 3. Preserve Educational Structure
- Keep timing information visible
- Maintain instructor guidance
- Preserve interactive elements
- Include assessment checkpoints

### Content Quality Standards

#### ✅ Essential Elements for Each Topic
- [ ] Clear learning objectives
- [ ] Structured lecture content with timing
- [ ] Instructor speaking notes
- [ ] Interactive elements or discussion prompts
- [ ] Visual aids where appropriate
- [ ] Progress tracking functionality
- [ ] Navigation to related content

#### ✅ Technical Requirements
- [ ] Proper relative path linking
- [ ] Responsive design compliance
- [ ] Semantic HTML structure
- [ ] Accessibility considerations
- [ ] Fast loading performance

## 🔧 Development Workflow

### Local Development
1. **Setup**: Use local server for development
2. **Testing**: Test all navigation paths and responsive design
3. **Content**: Add content following established patterns
4. **Validation**: Check all links and ensure proper styling

### Content Addition Process
1. **Plan Structure**: Define sessions, topics, and learning objectives
2. **Create Directories**: Follow established naming conventions
3. **Build Pages**: Use existing templates and component patterns
4. **Link Navigation**: Update week overview and main navigation
5. **Test**: Verify all paths and responsive behavior

### Style Customization
- **Colors**: Modify CSS variables in `assets/css/shared.css`
- **Typography**: Update font imports and CSS font families
- **Components**: Extend existing component classes
- **Animations**: Add to shared CSS for consistency

## 📋 Future Enhancements

### Content Features
- [ ] Video embedding support
- [ ] Interactive code examples
- [ ] Progress tracking with localStorage
- [ ] Printable course materials
- [ ] Search functionality
- [ ] Content filtering by difficulty

### Technical Features
- [ ] Dark/light mode toggle
- [ ] Offline functionality with service workers
- [ ] Progressive Web App (PWA) capabilities
- [ ] Analytics integration
- [ ] Comment system for discussions
- [ ] Auto-generated table of contents

## 🐛 Troubleshooting

### Common Issues

#### Styles Not Loading
- **Issue**: CSS not applying, default browser styles visible
- **Solution**: Use local server instead of file:// URLs
- **Command**: `python3 run-local-server.py`

#### Navigation Broken
- **Issue**: Links returning 404 or not found
- **Check**: Relative paths are correct (`../../assets/css/shared.css`)
- **Verify**: File structure matches expected hierarchy

#### Mobile Display Issues
- **Check**: Viewport meta tag is present
- **Test**: Use browser developer tools for mobile simulation
- **Verify**: CSS media queries are working

### Performance Optimization
- **Images**: Optimize any images you add
- **CSS**: Minify for production deployment
- **JavaScript**: Consider lazy loading for large content

## 📞 Support

### File Structure Reference
- **Main Hub**: `index.html`
- **Week Overview**: `weeks/week[N]/index.html`
- **Session Content**: `weeks/week[N]/session[N]-[N]/topic[N].html`
- **Projects**: `weeks/week[N]/project.html`
- **Shared Styles**: `assets/css/shared.css`
- **Shared Scripts**: `assets/js/navigation.js`

### Key CSS Classes
- `.content-section` - Main content containers
- `.lecture-section` - Individual lecture segments
- `.instructor-script` - Speaking notes for instructors
- `.interactive-element` - Hands-on activities
- `.highlight-box` - Important information callouts
- `.qa-section` - Question and answer segments

---

**Built with passion for AI education. Ready to scale from 1 week to 6+ weeks of comprehensive course content.**