# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive, hierarchical course website for a 6-week AI Engineering curriculum. Built as a static website optimized for GitHub Pages deployment with rich educational content and progressive course structure.

### Mission & Purpose
Transform traditional course materials into an interactive, navigable website featuring:
- **Hierarchical content organization** (Course → Weeks → Sessions → Topics)
- **Rich educational content** including lecture scripts, interactive elements, and instructor notes
- **Professional design** with Cyber Academic theme (dark mode with cyan/purple accents)
- **Mobile-responsive** design optimized for all devices
- **GitHub Pages ready** with zero external dependencies

### Target Content Structure
Expected markdown input files organized as:
```
/course-content/
  ├── week1.md - week6.md (course content by week)
  ├── course-overview.md (optional)
  └── resources.md (optional)
```

Each week's markdown should contain:
- Week title and overview
- Learning objectives
- Lecture details (title, topics, duration)
- Hands-on activities
- Project description and deliverables
- Resources and materials

## 🚨 IMPORTANT: Development Guide Updated

**For the latest development instructions, see `DEVELOPMENT.md`** - this file contains updated guidance for the refactored codebase with streamlined CSS/JS and template system.

## Quick Start & Development Commands

### Option 1: Local Development Server (Recommended)
```bash
cd ai-engineering-course
python3 run-local-server.py
# Opens http://localhost:8000 with cache-disabled headers for development
```

### Option 2: Simple Python Server
```bash
cd ai-engineering-course
python3 -m http.server 8000
# Opens http://localhost:8000
```

### Option 3: Direct File Access
Open `index.html` directly in browser (note: some features may not work due to CORS restrictions):
```
file:///path/to/ai-engineering-course/index.html
```

### Option 4: GitHub Pages Deployment
1. Create GitHub repository
2. Upload all files maintaining directory structure
3. Enable Pages in Settings > Pages
4. Select 'main' branch and '/ (root)' folder
5. Site will be live at: `https://username.github.io/repo-name/`

## Architecture & Code Structure

### Core Design System
- **Theme**: Cyber Academic (dark background with cyan/purple accents)
- **Typography**: Space Mono (display) + Inter Display (body)
- **Color Scheme**: CSS variables defined in `:root` of `assets/css/shared.css`
- **Layout**: CSS Grid and Flexbox, mobile-first responsive design

### Key File Locations
- **Main Hub**: `index.html` - Course overview and navigation
- **Shared Styles**: `assets/css/shared.css` - Unified CSS variables and components
- **Shared Scripts**: `assets/js/navigation.js` - Common JavaScript functionality
- **Week Content**: `weeks/week[N]/index.html` - Week overview pages
- **Session Hubs**: `weeks/week[N]/session[N]-[N]/index.html` - Session overview with topic navigation
- **Topic Content**: `weeks/week[N]/session[N]-[N]/topic[N].html` - Individual lesson pages
- **Project Pages**: `weeks/week[N]/project.html` - Project guides and deliverables
- **Source Content**: `course-content/` - Original markdown files

### Component System

#### CSS Classes for Content Structure
- `.content-section` - Main content containers
- `.lecture-section` - Individual lecture segments with timing
- `.instructor-script` - Speaking notes for instructors
- `.interactive-element` - Hands-on activities and discussion prompts
- `.content-box` - Slide content and structured information
- `.highlight-box` - Important callouts (`.info`, `.warning` variants)
- `.qa-section` - Question and answer segments
- `.instructor-notes` - Teaching tips and pacing information
- `.visual-aid` - Custom visual content containers

#### Session & Topic Navigation Components
- `.session-hero` - Session overview header with metadata
- `.topics-grid` - Grid layout for topic cards within sessions
- `.topic-card` - Individual topic navigation cards with objectives
- `.session-progress` - Progress tracking bar for session completion
- `.topic-navigation` - Previous/next navigation between topics
- `.ml-workflow-diagram` - Visual workflow representations
- `.algorithm-grid` - Algorithm overview card layouts

#### Navigation Components
- `.breadcrumb` - Hierarchical navigation paths (Course → Week → Session → Topic)
- `.progress-indicator` - Reading progress tracking
- Sticky navigation with backdrop blur effects

### Path Structure Conventions
- Relative paths from week content: `../../assets/css/shared.css`
- Relative paths from session index: `../../../assets/css/shared.css`
- Relative paths from topic content: `../../../../assets/css/shared.css`
- All paths relative to maintain GitHub Pages compatibility

### Navigation Hierarchy
The site follows a 4-level hierarchical structure:
1. **Course Level** (`index.html`) - Main course overview
2. **Week Level** (`weeks/week[N]/index.html`) - Week overview with session cards
3. **Session Level** (`weeks/week[N]/session[N]-[N]/index.html`) - Session overview with topic grid
4. **Topic Level** (`weeks/week[N]/session[N]-[N]/topic[N].html`) - Individual lesson content

## Content Management

### Adding New Content

#### For New Topics Within Existing Sessions
1. Create new topic file: `weeks/week[N]/session[N]-[N]/topic[X].html`
2. Copy template structure from existing topic files
3. Update session index page to include new topic card
4. Update navigation links in adjacent topics
5. Follow component class conventions for content structure

#### For New Sessions
1. Create session directory: `weeks/week[N]/session[N]-[N]/`
2. Create session index page with topic grid layout
3. Create individual topic files within session directory
4. Update week overview page to include new session card
5. Maintain proper breadcrumb and navigation structure

#### For New Weeks
1. Create week directory: `weeks/week[N]/`
2. Add week overview page using existing week templates
3. Update main course navigation in `index.html`
4. Create session subdirectories following naming conventions
5. Maintain consistent URL and linking structure

### Content Conversion Guidelines
- Convert markdown headers to appropriate HTML semantic structure
- Use instructor scripts for speaking notes and detailed explanations
- Wrap slide content in `.content-box` components
- Mark interactive elements with `.interactive-element` class
- Include timing information in `.lecture-timing` elements

## Technical Specifications

### Browser Compatibility
- Modern browsers (Chrome, Firefox, Safari, Edge latest versions)
- Mobile responsive design with tested breakpoints
- No external dependencies except Google Fonts

### Performance Considerations
- Inline critical CSS and JavaScript where appropriate
- Optimize for fast loading on GitHub Pages
- Cache-disabled headers in development server for testing

### GitHub Pages Deployment
- Static HTML/CSS/JavaScript only
- All paths relative for portability
- Root-level `index.html` serves as entry point
- Directory structure preserved during deployment

## Development Guidelines

### When Editing Content
- Maintain existing component class structure
- Preserve relative path relationships
- Follow established typography and color variable usage
- Test with local development server before deployment

### When Adding Features
- Extend existing CSS variables rather than hardcoding colors
- Add new component classes to shared.css for consistency
- Update navigation.js for shared JavaScript functionality
- Maintain mobile-first responsive design principles

## Educational Content Structure

The site converts educational markdown content into structured HTML with:
- Learning objectives as styled lists
- Lecture sections with timing and instructor guidance
- Interactive elements for hands-on activities
- Visual aids and timeline graphics
- Student Q&A sections and assessment checkpoints
- Comprehensive project guides with deliverables and evaluation criteria

## File Structure Reference

```
ai-engineering-course/
├── CLAUDE.md                              # This comprehensive guide
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
│       └── session1-1/                    # Session 1.1: Introduction to AI & ML
│           ├── index.html                 # Session overview with topic navigation
│           ├── topic1.html                # Topic 1: "What is AI?"
│           └── topic2.html                # Topic 2: "Machine Learning Fundamentals"
│
└── course-content/                        # Source markdown files
    ├── Week1_Session1.1_Topic1_What_is_AI.md           # Original content sources
    └── Week1_Session1.1_Topic2_Machine_Learning_Fundamentals.md
```

## AI Agent Implementation Instructions

### When Building Course Websites from Markdown

#### Step 1: Parse Markdown Files
1. Read all provided markdown files
2. Extract structured information:
   - Week number and title
   - Learning objectives (bullet points)
   - Lecture sessions (titles, topics, durations)
   - Hands-on activities
   - Project details and deliverables
   - Resources and links
3. Organize data into a structured format

#### Step 2: Design Philosophy
Follow Cyber Academic aesthetic direction:
- **Neon accents, dark theme, futuristic yet professional**
- **Typography**: Space Mono (display) + Inter Display (body)
- **Color Scheme**: Use established CSS variables
- **Animations**: Smooth transitions, hover effects, scroll triggers
- **Responsive**: Mobile-first approach
- **Semantic**: HTML5 elements

#### Step 3: Website Structure Options

**For Single-Page Implementation:**
1. **Hero Section** - Course title, stats, CTA
2. **Overview Section** - Description, highlights, timeline
3. **Curriculum Section** - Expandable week cards/accordions
4. **Features Section** - Learning approach, time commitment
5. **Footer** - Course info, contact, resources

**For Multi-Page Implementation (Current Architecture):**
- **Main Hub** (`index.html`) - Course overview with week cards
- **Week Pages** (`weeks/week[N]/index.html`) - Week overviews with session cards  
- **Session Hubs** (`weeks/week[N]/session[N]-[N]/index.html`) - Session overviews with topic navigation
- **Topic Pages** (`weeks/week[N]/session[N]-[N]/topic[N].html`) - Individual lesson content
- **Project Pages** (`weeks/week[N]/project.html`) - Project guides

#### Step 4: Content Integration Algorithm

```javascript
// Pseudo-code for processing markdown to HTML
function processMarkdownToHTML(weekMdFile) {
    const weekData = {
        number: extractWeekNumber(weekMdFile),
        title: extractTitle(weekMdFile),
        objectives: extractObjectives(weekMdFile),
        lecture1: extractLecture(weekMdFile, 1),
        lecture2: extractLecture(weekMdFile, 2),
        activities: extractActivities(weekMdFile),
        project: extractProject(weekMdFile),
        resources: extractResources(weekMdFile)
    };
    return generateWeekCard(weekData);
}
```

#### Step 5: Markdown Parsing Patterns

Recognize these patterns and convert appropriately:
```markdown
# Week 1: Title → <h2> for week titles
## Learning Objectives → <h3> for section headers
### Topics Covered → <h4> for subsections
- [ ] Deliverable → <input type="checkbox"> for deliverables
[text](url) → <a href="url">text</a>
**bold** → <strong>bold</strong>
`code` → <code>code</code>
```

#### Step 6: Quality Checklist

**Functionality:**
- [ ] Navigation works correctly
- [ ] All links functional
- [ ] Responsive design (mobile/tablet/desktop)
- [ ] No console errors

**Content:**
- [ ] All markdown accurately converted
- [ ] Sequential week numbering
- [ ] Learning objectives visible
- [ ] Projects and deliverables included
- [ ] Resources properly linked

**Design:**
- [ ] Consistent typography
- [ ] Cohesive color scheme
- [ ] Smooth animations
- [ ] Good contrast and readability
- [ ] Professional appearance

**Performance:**
- [ ] Fast loading
- [ ] Minimal external dependencies
- [ ] GitHub Pages compatible

#### Step 7: Error Handling

**Missing Content:**
- Use placeholder text: "[Content to be added]"
- Maintain structure with incomplete data
- Note missing sections in comments

**Inconsistent Format:**
- Parse flexibly with pattern matching
- Fallback to displaying raw content
- Maintain functionality even with errors

#### Step 8: Enhancement Opportunities

Consider adding:
- Search/filter functionality
- Progress tracking (localStorage)
- Print-friendly styles
- Dark/light mode toggle
- Interactive course timeline
- Downloadable PDF syllabus

### Priority Order for Implementation
1. **Functionality** - Everything works
2. **Content Accuracy** - Markdown correctly converted
3. **Visual Design** - Professional, distinctive appearance
4. **Enhancements** - Nice-to-have features

### Output Requirements
- **Single HTML file** OR **Multi-page structure** (depending on scope)
- **GitHub Pages compatible** (static only)
- **Fast loading** with minimal dependencies
- **Professional quality** suitable for sharing
- **Semantic code** for future editing