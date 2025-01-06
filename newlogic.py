import re

def extract_skills_from_jd(job_description):
    # Define sections and keywords to look for
    sections = ["Responsibilities", "Requirements", "Qualifications", "Skills", "Role Descriptions", "Basic Requirement", "Summary of Requirement"]
    keywords = ["proficient in", "experience with", "knowledge of", "ability to", "must have", "preferred", "required", "expectation", "support"]

    # Split the JD into lines for easier processing
    lines = job_description.split("\n")

    # Initialize lists to store the extracted skills and qualifications
    technical_skills = []
    qualifications = []
    soft_skills = []
    roles = []

    # Iterate over each line to extract relevant information
    current_section = None
    for line in lines:
        line = line.strip()

        # Check if the line is a section header
        if any(section.lower() in line.lower() for section in sections):
            current_section = line
            continue

        # Extract roles descriptions
        if current_section and "Role Descriptions" in current_section:
            roles.append(line)
            continue

        # Check if the line contains any of the keywords
        if any(keyword in line.lower() for keyword in keywords):
            if "degree" in line.lower() or re.search(r'\d+ years', line.lower()):
                qualifications.append(line)
            elif re.search(r'\b(led|significant role|experience in|domain experience|knowledge of|skills)\b', line.lower()):
                technical_skills.append(line)
            else:
                soft_skills.append(line)
        elif re.search(r'\b(problem-solving|attention to detail|collaborate|support|planning|analysis)\b', line.lower()):
            soft_skills.append(line)

    # Create a structured dictionary of the extracted information
    extracted_skills = {
        "Technical Skills": technical_skills,
        "Qualifications": qualifications,
        "Soft Skills": soft_skills,
        "Role Descriptions": roles
    }

    return extracted_skills

def clean_extracted_skills(extracted_skills):
    # Helper function to clean and refine the extracted skills
    def extract_keywords(text_list):
        skill_keywords = []
        for text in text_list:
            # Extract phrases within the text that match typical skill patterns
            phrases = re.findall(r'\b[A-Za-z0-9\s\(\)-]+\b', text)
            skill_keywords.extend(phrases)
        
        # Filter out common non-skill words and phrases
        filtered_keywords = []
        non_skill_words = {"the", "and", "with", "of", "in", "for", "to", "is", "on", "by", "at", "as", "a", "an", "or", "be", "from", "which", "will", "include", "including"}
        for keyword in skill_keywords:
            words = keyword.split()
            cleaned_words = [word for word in words if word.lower() not in non_skill_words]
            if cleaned_words:
                filtered_keywords.append(" ".join(cleaned_words))

        return list(set(filtered_keywords))

    cleaned_skills = {key: extract_keywords(value) for key, value in extracted_skills.items()}
    return cleaned_skills

# Example job description
job_description = """
Summary of Requirement:
Identify primary FSI / Banking skills in market indicated below with Domain and Digital Transformation experience.  Key roles include Project Manager and Business Analyst given the experience requirement below across levels of aConsultant, Senior Consultant, and Manager.  
Equivalent Positions for FTE Hiring	Years of Experience (Indicative)
Senior Consultant / Manager	9-10 Years
Senior Consultant	5-8 Years

Location / Markets	Bangalore

Basic Requirement:
•	Experience in Banking and Financial Services is a must.  Should have led / played significant role in end-to-end digital transformation programs and preferably in Agile environments
•	Domain experience required in any of the following Banking Operations:
o	Payments
o	Global Banking and Markets (Commercial Banking)
o	Wealth and Private Banking (Retail Banking)
	Payments:
•	International Payment Systems
•	Open-Banking and API driven payments
•	End-to-end (E2E) payment lifecycle experience
•	Corporate payments, cash management, and real-time/immediate payments
•	ISO 20022 migration experience
	Commercial Banking (CMB):
•	Corporate and cash management core products and services
•	Trade operation
•	Experience in Automated Data Flow project (RBI Automated reporting system)
•	Knowledge of regulatory related issues
	Retail Banking:
•	Retail lending, and investing concepts
•	Mortgage
•	Credit Card processes
•	Contact Center Optimization experience – IVR, Browser changes under Digital channels
•	Branch optimization

Role Descriptions:
Business Analyst (Manager (M1 band), Senior Consultant)
Associate will support analysis of problems and related design tasks. More exploratory analysis or design, in particular during the pre-initiation and/or initiation phases, or design work requiring specialist business or functional knowledge. The business analyst support will consist of activities in relation to the business requirements solution design, validation of solution, supporting UAT and change adoption for CLIENT programmes and / or projects, as follows:
•	Current state research, data gathering and analysis
•	Data analysis and modelling
•	Process analysis and documentation
•	Business requirements gathering from CLIENT stakeholders
•	Business requirements definition and prioritisation
•	Traceability of business requirements into solution design to confirm linkage of business functionality/expectations in any new proposed solution
•	High level business solution design, which may cover aspects of customer journey design, policy design, business process design, business procedure design, organisation/people design and high-level functional systems design (for example, architecture, use cases)
•	Evaluation of relative costs, benefits and obstacles of potential solutions
•	Traceability of requirements and solution design into subsequent system design activity, through liaison with the CLIENT Software Delivery team in order to confirm technical designs follow the business design
•	Analysis to support key decisions made by the Project sponsor or steering committee
•	Align the analysis effort to SAFe methodology.

In addition to specific expectation from business analyst, for specific projects the overall analysis effort will be divided in two categories (a) Market Business Analysis and (b) Digital Business Analysis. Market BA will primarily focus on analyzing business challenges & identifying relevant business requirement for digitization keeping in mind the overall programme objectives. It is expected that the associate is exposed to digital project managed in agile fashion (CSM certified, SAFe methodology) and has good communication skills and exposed to requirement gathering, managing backlogs, generating business case & stakeholder mgt and support market adoption effort. Digital BA would further engage with technology and other key stakeholders to agree on suitable solution that meet the specific business need. Associate is expected to have understanding of specific digital tools, integration approaches to help conclude the effort swiftly. Associate will also take care of actual solution validation & provide support during execution and rollout stages.

Project Manager (Manager, Experienced Senior Consultant level candidate) 
Associate playing project manager role is expected to manage the specific project using SAFe methodology. The project management for projects or programmes will consist of activities in relation to planning and monitoring as follows:
•	Definition of programme / project scope and goals
•	Planning support which will cover approach definition, work breakdown, sequencing and milestone definition
•	Estimation of resource effort and cost budgets
•	Resource planning
•	Identification of project risks and issues
•	Tracking mitigating actions in relation to programme / project risks and issues
•	Identification of dependencies internal to the programme / project and externally to other projects and business areas
•	Support all essential governance activities including working group & steering committee through documentation including meeting minutes and recording of agreed actions
•	Task allocation across teams and team members
•	Tracking against activity or milestone completion and escalate variance to CLIENT programme / project plan (in accordance with CLIENT’s agreed escalation process)
•	Project status reporting (where required)
•	Support readiness decisions and follow up actions from programme / project tollgates

"""

# Extract skills from the job description
extracted_skills = extract_skills_from_jd(job_description)

# Clean and refine the extracted skills
cleaned_skills = clean_extracted_skills(extracted_skills)

# Print the cleaned and extracted skills
print("Technical Skills:")
for skill in cleaned_skills["Technical Skills"]:
    print(f"- {skill}")

print("\nQualifications:")
for qualification in cleaned_skills["Qualifications"]:
    print(f"- {qualification}")

print("\nSoft Skills:")
for skill in cleaned_skills["Soft Skills"]:
    print(f"- {skill}")

print("\nRole Descriptions:")
for role in cleaned_skills["Role Descriptions"]:
    print(f"- {role}")
