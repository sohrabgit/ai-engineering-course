# AI Agent Instructions: Building AI Engineering Course Website

## Mission
You are tasked with creating a professional, interactive course website for a 6-week AI Engineering curriculum using provided markdown files. The website will be deployed on GitHub Pages using static HTML/CSS/JavaScript.

## Input Files You'll Receive
The user will provide markdown (.md) files organized as follows:
```
/course-content/
  ├── week1.md (only one topic provided. will add content later)
  ├── week2.md (will provide later)
  ├── week3.md (will provide later)
  ├── week4.md (will provide later)
  ├── week5.md (will provide later)
  ├── week6.md (will provide later)
  ├── course-overview.md (optional)
  └── resources.md (optional)
```

## Expected Markdown Structure
Each week's markdown file should contain:
- Week title and overview
- Learning objectives
- Lecture 1 details (title, topics, duration)
- Lecture 2 details (title, topics, duration)
- Hands-on activities
- Project description and deliverables
- Resources and materials

## Your Implementation Steps

### STEP 1: Parse Markdown Files
1. Read all provided markdown files
2. Extract structured information:
   - Week number and title
   - Learning objectives (bullet points)
   - Lecture sessions (titles, topics, durations)
   - Hands-on activities
   - Project details and deliverables
   - Resources and links
3. Organize data into a structured JSON or JavaScript object format

### STEP 2: Design Philosophy
Follow these design principles from the frontend-design skill:

**Aesthetic Direction**: Choose ONE bold direction:
- Option A: **Tech Brutalism** - Raw, geometric, high contrast, monospace accents
- Option B: **Cyber Academic** - Neon accents, dark theme, futuristic yet professional
- Option C: **Minimal Editorial** - Clean, magazine-style, excellent typography, generous whitespace
- Option D: **Gradient Maximalism** - Bold colors, overlapping elements, dynamic effects

**Key Requirements**:
- Use distinctive typography (NOT Inter, Arial, Roboto)
  - Display font: Syne, Space Mono, Anybody, Outfit, or similar distinctive choice
  - Body font: Epilogue, Inter Display, Public Sans, or refined alternative
- Create a cohesive color scheme with CSS variables
- Add meaningful animations (staggered reveals, hover effects, scroll triggers)
- Implement responsive design (mobile-first approach)
- Use semantic HTML5 elements

### STEP 3: Website Structure
Create a single-page website with these sections:

```html
1. HERO SECTION
   - Course title: "AI Engineering Course"
   - Subtitle: "6-Week Intensive Curriculum"
   - Key stats: Duration, Hours/week, Week count
   - CTA button: "Explore Curriculum" (smooth scroll to weeks)

2. OVERVIEW SECTION
   - Course description
   - What students will learn
   - Course highlights/features
   - Visual timeline or progress indicator

3. CURRICULUM SECTION (Main Content)
   For each week, create an expandable card/accordion:
   
   WEEK CARD:
   - Week number and title (collapsible header)
   - Learning objectives (when expanded)
   - Two lecture sessions with:
     * Lecture title
     * Topics covered (list)
     * Duration (90 minutes)
     * Hands-on activities
   - Project section:
     * Project description
     * Deliverables (checklist style)
   - Resources section:
     * Links to materials
     * Additional readings

4. COURSE FEATURES SECTION
   - Hands-on learning emphasis
   - Progressive difficulty
   - Real-world applications
   - Capstone project
   - Time commitment (4-6 hours/week)

5. FOOTER
   - Course information
   - Contact/enrollment info
   - Links to resources
   - Copyright notice
```

### STEP 4: Technical Implementation

**HTML Structure**:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Engineering Course - 6-Week Curriculum</title>
    <!-- Google Fonts -->
    <!-- Inline CSS -->
</head>
<body>
    <!-- Navigation (sticky) -->
    <!-- Hero Section -->
    <!-- Overview Section -->
    <!-- Curriculum Section -->
    <!-- Features Section -->
    <!-- Footer -->
    <!-- Inline JavaScript -->
</body>
</html>
```

**CSS Requirements**:
- Use CSS Grid and Flexbox for layouts
- Implement CSS variables for theming:
```css
:root {
    --primary-color: #...;
    --accent-color: #...;
    --background: #...;
    --surface: #...;
    --text-primary: #...;
    --text-secondary: #...;
}
```
- Add smooth transitions and animations:
  - Fade-in on scroll (intersection observer)
  - Staggered reveals for list items
  - Hover effects on cards
  - Smooth accordion expand/collapse
- Responsive breakpoints: 768px (tablet), 1024px (desktop)

**JavaScript Features**:
```javascript
// 1. Smooth scrolling for navigation
// 2. Accordion/collapse functionality for week cards
// 3. Intersection Observer for scroll animations
// 4. Progress indicator (optional)
// 5. "Back to top" button (optional)
// 6. Week filter/search (optional enhancement)
```

### STEP 5: Content Integration Algorithm

```javascript
// Pseudo-code for processing markdown to HTML
function processMarkdownToHTML(weekMdFile) {
    // Parse markdown sections
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
    
    // Generate HTML card
    return generateWeekCard(weekData);
}

function generateWeekCard(weekData) {
    return `
        <div class="week-card" data-week="${weekData.number}">
            <div class="week-header">
                <h3>Week ${weekData.number}: ${weekData.title}</h3>
                <button class="toggle-btn">▼</button>
            </div>
            <div class="week-content">
                ${generateObjectives(weekData.objectives)}
                ${generateLecture(weekData.lecture1, 1)}
                ${generateLecture(weekData.lecture2, 2)}
                ${generateActivities(weekData.activities)}
                ${generateProject(weekData.project)}
                ${generateResources(weekData.resources)}
            </div>
        </div>
    `;
}
```

### STEP 6: Markdown Parsing Patterns

Recognize these common markdown patterns:

```markdown
# Week 1: Title
## Learning Objectives
- Objective 1
- Objective 2

## Lecture 1: Lecture Title (90 minutes)
### Topics Covered
- Topic 1
- Topic 2

### Hands-on Activities
- Activity description

## Lecture 2: Lecture Title (90 minutes)
...

## Project: Project Name
### Description
Project details here

### Deliverables
- [ ] Deliverable 1
- [ ] Deliverable 2

## Resources
- [Resource Name](URL)
- Reading material
```

Convert to appropriate HTML:
- `#` → `<h2>` for week titles
- `##` → `<h3>` for section headers
- `###` → `<h4>` for subsections
- `- [ ]` → `<input type="checkbox">` for deliverables
- `[text](url)` → `<a href="url">text</a>`
- `**bold**` → `<strong>bold</strong>`
- `` `code` `` → `<code>code</code>`

### STEP 7: Quality Checklist

Before delivering the final HTML file, verify:

**Functionality**:
- [ ] All week cards expand/collapse correctly
- [ ] Smooth scrolling works for navigation
- [ ] All links are functional
- [ ] Responsive design works on mobile, tablet, desktop
- [ ] No console errors in browser

**Content**:
- [ ] All markdown content is accurately converted
- [ ] Week numbers are sequential (1-6)
- [ ] Learning objectives are clearly listed
- [ ] Both lectures are included for each week
- [ ] Projects and deliverables are visible
- [ ] Resources are properly linked

**Design**:
- [ ] Consistent typography throughout
- [ ] Cohesive color scheme
- [ ] Smooth animations (not jarring)
- [ ] Good contrast for readability
- [ ] Proper spacing and alignment
- [ ] Professional, polished appearance

**Performance**:
- [ ] Single HTML file (all CSS/JS inline)
- [ ] Optimized for fast loading
- [ ] No external dependencies (except fonts)
- [ ] Works offline after initial load

### STEP 8: Output Format

Deliver a single file named: `index.html`

File structure:
```html
<!DOCTYPE html>
<html>
<head>
    <!-- All CSS inline in <style> tags -->
</head>
<body>
    <!-- All HTML content -->
    
    <!-- All JavaScript inline in <script> tags at end -->
</body>
</html>
```

### STEP 9: User Instructions to Include

At the top of the HTML file (in comments), include:

```html
<!--
AI ENGINEERING COURSE WEBSITE
Generated: [DATE]

DEPLOYMENT INSTRUCTIONS:
1. Save this file as 'index.html'
2. Create a GitHub repository
3. Upload this file to the repository
4. Enable GitHub Pages in Settings > Pages
5. Select 'main' branch and '/ (root)' folder
6. Your site will be live at: https://username.github.io/repo-name/

CUSTOMIZATION:
- Edit CSS variables in the :root section to change colors
- Modify content directly in HTML sections
- Add your own images by uploading to repo and updating src paths

BROWSER COMPATIBILITY:
- Chrome, Firefox, Safari, Edge (latest versions)
- Mobile responsive
-->
```

## Example Interaction Flow

**User provides**:
```
Here are my course files:
- week1.md
- week2.md
- week3.md
- week4.md
- week5.md
- week6.md
```

**You respond**:
1. "I'll create your AI Engineering course website. Let me analyze your markdown files..."
2. Read and parse all 6 markdown files
3. Extract structured data from each
4. Choose an aesthetic direction (state your choice and why)
5. Generate the complete HTML file
6. "I've created your course website with [aesthetic choice]. The site includes all 6 weeks with expandable content, smooth animations, and responsive design. Here's your index.html file..."
7. Present the file for download
8. Provide deployment instructions

## Error Handling

If markdown files are missing sections:
- Use placeholder text: "[Content to be added]"
- Maintain structure even with incomplete data
- Note missing sections in comments

If markdown format is inconsistent:
- Attempt to parse flexibly
- Use intelligent pattern matching
- Fallback to displaying raw content if necessary

## Enhancement Opportunities

If time/complexity allows, consider adding:
- Search/filter functionality for weeks
- Print-friendly stylesheet
- Progress tracking (with localStorage)
- Dark/light mode toggle
- Downloadable PDF syllabus
- Interactive course timeline
- Testimonials section (if data provided)

## Final Notes

**Priority Order**:
1. Functionality (everything works)
2. Content accuracy (markdown correctly converted)
3. Visual design (professional, distinctive)
4. Enhancements (nice-to-have features)

**Remember**:
- Single HTML file for easy deployment
- GitHub Pages compatible (static only)
- Fast loading, no external dependencies
- Professional quality suitable for sharing
- Clear, semantic code for future editing

Good luck building an exceptional AI Engineering course website!
