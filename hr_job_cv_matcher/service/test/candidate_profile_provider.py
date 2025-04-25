
from hr_job_cv_matcher.service.test.job_description_cv_provider import CV
from langchain.schema import Document
from hr_job_cv_matcher.model import CandidateProfile, EducationCareerJson, MatchSkillsProfileJson


def create_candidate_profile() -> CandidateProfile:
    source = "dummy.odf"
    return CandidateProfile(
        source=source,
        document=Document(page_content=CV, metadata={"source": source}),
        matched_skills_profile=MatchSkillsProfileJson(
            matching_skills=["Wordpress", "PHP", "HTML"],
            matching_skills_description="The candidate has 3 years of experience with these technologies. Has built multiple websites using WordPress and PHP.",
            missing_skills=["Javascript", "Phigma"],
            missing_skills_recommendations="Recommended to take online courses in JavaScript and Figma to meet all job requirements.",
            social_skills=["Managing teams"],
            social_skills_description="Has led development teams of 5-10 people. Demonstrated strong communication skills in client interactions.",
            candidate_summary="Strong technical background in web development with room for growth in modern JavaScript frameworks."
        ),
        education_career_profile=EducationCareerJson(
            relevant_job_list=["Front End Developer", "Wordpress Administrator"],
            relevant_degree_list=["B. Tech"],
            years_of_experience=3
        ),
        score=0,
        breakdown=""
    )


if __name__ == "__main__":
    from hr_job_cv_matcher.log_init import logger
    logger.info("Candidate profile: %s", create_candidate_profile())
