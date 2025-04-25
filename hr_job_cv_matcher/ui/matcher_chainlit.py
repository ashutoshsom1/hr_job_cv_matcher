from pathlib import Path
from asyncer import asyncify
from hr_job_cv_matcher.list_utils import convert_list_to_markdown
from hr_job_cv_matcher.model import (
    CandidateProfile,
    EducationCareerJson,
    MatchSkillsProfileJson,
    ScoreWeightsJson,
)
from hr_job_cv_matcher.service.candidate_ranking import (
    DEFAULT_WEIGHTS,
    calculate_score,
    sort_candidates,
)
from hr_job_cv_matcher.service.chart_generator import generate_chart
from hr_job_cv_matcher.service.job_description_cv_matcher import (
    MatchSkillsProfile,
    create_input_list,
    create_match_profile_chain_pydantic,
)
from hr_job_cv_matcher.service.education_extraction import (
    create_education_chain,
)
from hr_job_cv_matcher.service.social_skills_extractor import (
    create_social_profile_chain,
    extract_social_skills,
)
from hr_job_cv_matcher.ui.chat_settings import create_chat_settings
from hr_job_cv_matcher.ui.messages import (
    display_uploaded_job_description,
    render_barchart_image,
)
from langchain import LLMChain
from langchain.schema import Document
from typing import List, Dict, Tuple, Optional

import chainlit as cl

from hr_job_cv_matcher.document_factory import convert_to_doc

from hr_job_cv_matcher.log_init import logger
from hr_job_cv_matcher.config import cfg, prompt_cfg

TIMEOUT = 1200
LLM_AUTHOR = "LLM"
HR_ASSISTANT = "HR Assistant"

KEY_APPLICATION_DOCS = "application_docs"
KEY_CV_DOCS = "cv_docs"


@cl.on_chat_start
async def init():
    await cl.Message(
        content=f"**May the work-force be with you.**",
    ).send()
    application_docs = await upload_and_extract_text(
        "job description files", max_files=cfg.max_jd_files
    )

    if application_docs is not None and len(application_docs) > 0:
        cv_docs = await upload_and_extract_text("CV files", max_files=cfg.max_cv_files)
        if application_docs and cv_docs:
            await start_process_applications_and_cvs(application_docs, cv_docs)

        else:
            await cl.ErrorMessage(
                content=f"Could not process the CVs. Please try again",
            ).send()
    else:
        await cl.ErrorMessage(
            content=f"Could not process the application document. Please try again",
        ).send()


@cl.on_settings_update
async def process_with_settings(settings):
    logger.info("Settings: %s", settings)
    application_docs: List[Document] = cl.user_session.get(KEY_APPLICATION_DOCS)
    cvs_docs = cl.user_session.get(KEY_CV_DOCS)
    source_document_dict: Dict[str, Document] = {
        Path(doc.metadata["source"]).name: doc for doc in application_docs
    }
    logger.info("application_docs: %s", len(application_docs))
    score_weights = ScoreWeightsJson.factory(settings)
    prompt_cfg.update_prompt(settings)
    candidate_profiles_dict: Dict[
        str, List[CandidateProfile]
    ] = await process_applications_and_cvs(application_docs, cvs_docs, score_weights)
    sorted_candidate_profiles_dict: Dict[str, List[CandidateProfile]] = await asyncify(
        sort_candidates
    )(candidate_profiles_dict)
    await render_ranking(sorted_candidate_profiles_dict, source_document_dict)


async def render_ranking(
    sorted_candidate_profiles_dict: Dict[str, List[CandidateProfile]],
    source_document_dict: Dict[str, Document],
):
    for jd_source, sorted_candidate_profiles in sorted_candidate_profiles_dict.items():
        doc = source_document_dict[jd_source]
        await display_uploaded_job_description(doc)
        
        ranking_message = "## Ranking\n"
        elements = []
        
        for i, profile in enumerate(sorted_candidate_profiles):
            path = Path(profile.source)
            
            # Check if file exists in original location
            if not path.exists():
                # Try to find the file in the temp directory
                temp_path = Path(cfg.temp_doc_location) / path.name
                if temp_path.exists():
                    path = temp_path
                else:
                    # If file not found, just show the name without link
                    ranking_message += f"\n{i + 1}. {path.stem}: **{profile.score}** points"
                    logger.warning(f"PDF file not found: {path}")
                    continue
            
            try:
                # Add PDF element for each candidate
                elements.append(
                    cl.Pdf(
                        name=f"cv_{i+1}",
                        display="inline",
                        path=str(path.absolute())
                    )
                )
                # Create markdown link to the PDF
                ranking_message += f"\n{i + 1}. [{path.stem}](click:cv_{i+1}): **{profile.score}** points"
            except Exception as e:
                # If there's any error adding the PDF, just show the name without link
                ranking_message += f"\n{i + 1}. {path.stem}: **{profile.score}** points"
                logger.error(f"Error adding PDF {path}: {str(e)}")

        # Only send elements if we have any
        if elements:
            await cl.Message(
                content=ranking_message,
                elements=elements
            ).send()
        else:
            await cl.Message(content=ranking_message).send()

        breakdown_msg_id = await cl.Message(content=f"### Breakdown").send()
        await render_breakdown(sorted_candidate_profiles, breakdown_msg_id)


async def render_breakdown(
    sorted_candidate_profiles: List[CandidateProfile], breakdown_msg_id: str
):
    for profile in sorted_candidate_profiles:
        # Get candidate name from the source file
        candidate_name = Path(profile.source).stem
        
        # Create a header for each candidate
        await cl.Message(
            content=f"## 📋 Candidate Profile: {candidate_name}",
            author=HR_ASSISTANT,
            parent_id=breakdown_msg_id
        ).send()

        # Render PDF
        await render_pdf(profile, breakdown_msg_id)
        
        skills = profile.matched_skills_profile
        if skills:
            # Skills Analysis Section
            skills_header = f"""### 🎯 Skills Analysis
            """
            msg_id = await cl.Message(
                content=skills_header,
                author=HR_ASSISTANT,
                parent_id=breakdown_msg_id
            ).send()
            
            await render_scoring(skills, msg_id)
            await render_skills(skills, msg_id=msg_id)

        education_career_profile = profile.education_career_profile
        if education_career_profile:
            # Education & Experience Section
            education_header = f"""### 📚 Education & Experience
            """
            edu_msg_id = await cl.Message(
                content=education_header,
                author=HR_ASSISTANT,
                parent_id=breakdown_msg_id
            ).send()
            
            await render_education(education_career_profile, edu_msg_id)

        if skills and education_career_profile:
            # Overall Assessment Section
            assessment_header = f"""### 📊 Overall Assessment
            """
            assessment_msg_id = await cl.Message(
                content=assessment_header,
                author=HR_ASSISTANT,
                parent_id=breakdown_msg_id
            ).send()
            
            await render_score(profile, assessment_msg_id)

        # Add a separator between candidates
        await cl.Message(
            content="---",
            author=HR_ASSISTANT,
            parent_id=breakdown_msg_id
        ).send()


async def process_applications_and_cvs(
    application_docs: List[Document],
    cvs_docs: List[Document],
    score_weights: ScoreWeightsJson,
) -> Dict[str, List[CandidateProfile]]:
    await cl.Message(content=f"{len(cvs_docs)} CV(s) uploaded.").send()
    jd_cv_result_matrix = {}
    for current_application_doc in application_docs:
        candidate_profiles: List[
            CandidateProfile
        ] = await process_job_description_and_candidates(
            current_application_doc, cvs_docs
        )
        scored_profiles = []
        if len(candidate_profiles) > 0:
            logger.warn("%d profiles extracted", len(candidate_profiles))
            for profile in candidate_profiles:
                skills = profile.matched_skills_profile
                education_career_profile = profile.education_career_profile
                if skills and education_career_profile:
                    score, breakdown = calculate_score(profile, score_weights)
                    profile.score = score
                    profile.breakdown = breakdown
                    scored_profiles.append(profile)
        else:
            logger.warn("No profiles extracted")
        jd_cv_result_matrix[
            Path(current_application_doc.metadata["source"]).name
        ] = candidate_profiles
    return jd_cv_result_matrix


async def upload_and_extract_text(
    item_to_upload: str, max_files: int = 1
) -> List[Document]:
    files = None
    application_docs: List[Document] = []
    while files is None:
        files = await cl.AskFileMessage(
            content=f"Please upload {item_to_upload}!",
            accept=["application/pdf"],
            max_files=max_files,
            timeout=TIMEOUT,
        ).send()
        if files is not None:
            for file in files:
                logger.info(type(file))
                await cl.Message(
                    content=f"Processing {file.name}. Please wait ...",
                ).send()
                application_docs.append(await asyncify(convert_to_doc)(file=file))
                logger.info("Document: %s", application_docs)
    return application_docs


def clean_skills_lists(matching_skills: List[str], missing_skills: List[str]) -> Tuple[List[str], List[str]]:
    """Remove any skills that appear in both matching and missing skills lists."""
    # Convert to sets for easier comparison
    matching_set = {skill.lower().strip() for skill in matching_skills}
    missing_set = {skill.lower().replace(' (desired)', '').strip() for skill in missing_skills}
    
    # Remove any skills from missing_set that are already in matching_set
    cleaned_missing = [skill for skill in missing_skills 
                      if skill.lower().replace(' (desired)', '').strip() not in matching_set]
    
    return matching_skills, cleaned_missing


async def process_job_description_and_candidates(
    application_doc: Document, cv_documents: List[Document]
) -> List[CandidateProfile]:
    sources, input_list = extract_sources_input_list(application_doc, cv_documents)
    profile_llm_chain = create_match_profile_chain_pydantic()
    social_profile_llm_chain = create_social_profile_chain()

    _, skill_results = await process_skills_llm_chain(
        input_list, profile_llm_chain, sources
    )
    _, social_skill_results = await process_social_skills_llm_chain(
        input_list, social_profile_llm_chain, sources
    )
    education_results = await process_career_llm_chain(input_list, sources)

    extracted_profiles: List[CandidateProfile] = []

    for source in skill_results.keys():
        skill_result = skill_results[source]
        education_result = education_results[source]
        social_skill_result = social_skill_results[source]
        match_skills_profile = None

        try:
            if "function" in skill_result:
                match_skills_profile: MatchSkillsProfile = skill_result["function"]
            elif isinstance(skill_result, MatchSkillsProfile):
                match_skills_profile: MatchSkillsProfile = skill_result
            else:
                logger.warning(f"Unexpected skill_result format: {type(skill_result)}")
                continue
            
            if match_skills_profile is None:
                logger.warning("No match_skills_profile found")
                continue
        except Exception as e:
            logger.error(f"Error processing skill result: {e}")
            continue

        logger.info("Matching skills: %a", match_skills_profile)

        # Process social skills
        social_kills = extract_social_skills(social_skill_result)

        # Create default descriptions if not available
        matching_skills_desc = getattr(match_skills_profile, 'matching_skills_description', 
            "The candidate has demonstrated proficiency in these skills through their work experience.")
        missing_skills_rec = getattr(match_skills_profile, 'missing_skills_recommendations',
            "Consider pursuing training or certification in these areas to enhance qualifications.")
        social_skills_desc = getattr(match_skills_profile, 'social_skills_description',
            "These social skills were demonstrated through various professional interactions and team projects.")
        candidate_summary = getattr(match_skills_profile, 'candidate_summary',
            f"The candidate matches {len(match_skills_profile.matching_skills)} required skills with {len(match_skills_profile.missing_skills)} areas for development.")

        # Clean up duplicate skills
        cleaned_matching, cleaned_missing = clean_skills_lists(
            match_skills_profile.matching_skills,
            match_skills_profile.missing_skills
        )
        match_skills_profile.matching_skills = cleaned_matching
        match_skills_profile.missing_skills = cleaned_missing

        # Process career
        education_career_dict = None
        if "function" in education_result:
            education_career_dict: dict = education_result["function"]
        elif "relevant_job_list" in education_result:
            education_career_dict: dict = education_result
        if education_career_dict is None:
            continue

        logger.info("Matching education: %a", education_career_dict)
        education_career_json = EducationCareerJson(
            relevant_degree_list=education_career_dict["relevant_degree_list"],
            relevant_job_list=education_career_dict["relevant_job_list"],
            years_of_experience=education_career_dict["years_of_experience"],
        )
        extracted_profiles.append(
            CandidateProfile(
                source=source,
                document=application_doc,
                matched_skills_profile=MatchSkillsProfileJson(
                    matching_skills=match_skills_profile.matching_skills,
                    matching_skills_description=matching_skills_desc,
                    missing_skills=match_skills_profile.missing_skills,
                    missing_skills_recommendations=missing_skills_rec,
                    social_skills=social_kills,
                    social_skills_description=social_skills_desc,
                    candidate_summary=candidate_summary
                ),
                education_career_profile=education_career_json,
                score=0,
                breakdown="",
            )
        )

    if not (len(sources) == len(skill_results) == len(education_results)):
        await cl.ErrorMessage(
            content=f"The number of sources {len(sources)} and results (skills: {len(skill_results)}, education: {len(education_results)}) is not the same",
        ).send()
    return extracted_profiles


async def process_career_llm_chain(input_list, sources) -> dict:
    education_llm_chain = create_education_chain()
    education_results_msg = cl.Message(content="")
    
    _, education_results = await process_generic_extraction(
        input_list,
        education_llm_chain,
        sources,
        education_results_msg,
        "career"
    )
    
    return education_results


async def process_generic_extraction(
    input_list: List[Dict],
    llm_chain: LLMChain,
    sources: List[str],
    msg_prefix: str = "",
    author: str = "LLM"
) -> Tuple[str, Dict]:
    source_paths = [Path(source) for source in sources]
    source_dict = {}
    
    for input_item, source in zip(input_list, source_paths):
        source_name = source.name
        await cl.Message(
            content=f"{msg_prefix} Analyzing {source_name} 🔄",
            author=author
        ).send()
        try:
            result = await asyncify(llm_chain.predict)(
                **input_item
            )
            source_dict[source_name] = result
        except Exception as e:
            logger.error(f"Error processing {source_name}: {e}")
            source_dict[source_name] = None

    return msg_prefix, source_dict


async def process_skills_llm_chain(
    input_list: List[Dict],
    profile_llm_chain: LLMChain,
    sources: List[str]
) -> Tuple[str, Dict]:
    return await process_generic_extraction(
        input_list,
        profile_llm_chain,
        sources,
        "Skills extraction"  # msg_prefix
    )


async def process_social_skills_llm_chain(
    input_list: List[str], llm_chain: LLMChain, sources: List[str]
) -> Tuple[cl.Message, dict]:
    return await process_generic_extraction(
        input_list, llm_chain, sources, {"cv": "'CV'"}, "social skill"
    )


def extract_sources_input_list(
    application_doc, cv_documents
) -> Tuple[List[str], List[Dict]]:
    job_description = application_doc.page_content
    cvs = [c.page_content for c in cv_documents]
    sources = [c.metadata["source"] for c in cv_documents]
    input_list = create_input_list(job_description, cvs)
    return sources, input_list


async def render_skills(match_skills_profile: MatchSkillsProfileJson, msg_id: str):
    # Matching Skills Section
    matching_skills_content = "### Matching Skills\n"
    for skill in match_skills_profile.matching_skills:
        matching_skills_content += f"- {skill}\n"
    matching_skills_content += f"\n**Experience:**\n{match_skills_profile.matching_skills_description}\n"
    await cl.Message(content=matching_skills_content, author=LLM_AUTHOR, parent_id=msg_id).send()
    
    # Missing Skills Section
    missing_skills_content = "### Missing Skills\n"
    for skill in match_skills_profile.missing_skills:
        missing_skills_content += f"- {skill}\n"
    missing_skills_content += f"\n**Recommendations:**\n{match_skills_profile.missing_skills_recommendations}\n"
    await cl.Message(content=missing_skills_content, author=LLM_AUTHOR, parent_id=msg_id).send()
    
    # Social Skills Section
    social_skills_content = "### Social Skills\n"
    for skill in match_skills_profile.social_skills:
        social_skills_content += f"- {skill}\n"
    social_skills_content += f"\n**Demonstrated Through:**\n{match_skills_profile.social_skills_description}\n"
    await cl.Message(content=social_skills_content, author=LLM_AUTHOR, parent_id=msg_id).send()
    
    # Overall Summary
    summary_content = "### Candidate Summary\n" + match_skills_profile.candidate_summary
    await cl.Message(content=summary_content, author=LLM_AUTHOR, parent_id=msg_id).send()


async def render_education(
    education_career: EducationCareerJson, breakdown_msg_id: str
):
    try:
        relevant_degree_list = education_career.relevant_degree_list
        degree_output = convert_list_to_markdown(relevant_degree_list)
        relevant_job_list = education_career.relevant_job_list
        job_output = convert_list_to_markdown(relevant_job_list)
        logger.info("years_of_experience: %s", education_career.years_of_experience)
        years_of_experience = (
            ""
            if education_career.years_of_experience is None
            else f"- Years of experience: {education_career.years_of_experience}"
        )

        if len(degree_output) > 0 or len(job_output) > 0:
            detailed_output = f"""
#### Degrees
{degree_output}
#### Jobs
{job_output}
"""
            summary = f"""
#### Experience:
- Relevant jobs: {len(relevant_job_list)}
- Relevant degrees: {len(relevant_degree_list)}
{years_of_experience}
"""
            msg_id = await cl.Message(
                content=summary, author=LLM_AUTHOR, parent_id=breakdown_msg_id
            ).send()
            await cl.Message(
                content=detailed_output, author=LLM_AUTHOR, parent_id=msg_id
            ).send()
    except:
        logger.exception("Could not render education")


async def render_pdf(candidate_profile: CandidateProfile, breakdown_msg_id: str):
    source = candidate_profile.source
    path = Path(source)
    if path.exists():
        elements = [cl.Pdf(name=path.stem, display="inline", path=str(path.absolute()))]
        await cl.Message(
            content=f"#### {path.name}",
            author=HR_ASSISTANT,
            parent_id=breakdown_msg_id,
            elements=elements,
        ).send()


async def render_scoring(
    match_skills_profile: MatchSkillsProfileJson, breakdown_msg_id: str
):
    matching_skills_count = len(match_skills_profile.matching_skills)
    missing_skills_count = len(match_skills_profile.missing_skills)
    social_skills_count = len(match_skills_profile.social_skills)
    message = f"""
#### Skills:
- Matching skills: {matching_skills_count}
- Missing skills: {missing_skills_count}
- Social skills: {social_skills_count}
"""
    return await cl.Message(
        content=message, author=LLM_AUTHOR, parent_id=breakdown_msg_id
    ).send()


async def render_score(profile: CandidateProfile, breakdown_msg_id: str):
    message = f"""
#### Score:
- **{profile.score}**
- {profile.breakdown}
"""
    return await cl.Message(
        content=message, author=LLM_AUTHOR, parent_id=breakdown_msg_id
    ).send()


def render_skills_str(title: str, skills: List[Dict]) -> str:
    matching_skill_str = f"### {title}"
    for matching_skill in skills:
        matching_skill_str += f"\n- {matching_skill}"
    return matching_skill_str


async def start_process_applications_and_cvs(
    application_docs: List[Document], cvs_docs: List[Document]
):
    cl.user_session.set(KEY_APPLICATION_DOCS, application_docs)
    cl.user_session.set(KEY_CV_DOCS, cvs_docs)
    chat_settings = create_chat_settings()
    res: Optional[dict] = None
    settings = None
    while True:
        res = await cl.AskUserMessage(
            content="Now you can change the weights via the settings button. Please type 'ok' to process all applications and CVs.",
            timeout=TIMEOUT,
        ).send()
        if res is not None and "ok" in res["content"].lower():
            settings = await chat_settings.send()
            break
    await process_with_settings(settings)
