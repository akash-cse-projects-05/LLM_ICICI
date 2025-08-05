# Insurance Policy Dataset

## Overview

This repository contains a structured dataset derived from insurance policy documents, including sections such as the Preamble, Definitions, Terms & Conditions, and Exclusions. The dataset is designed to facilitate the development of software applications such as Customer Relationship Management (CRM) systems, insurance chatbots, and other tools that require understanding and processing of insurance policies.

The dataset is created based on the Disclosure to Information Norm and regulatory guidelines specified by IRDAI (Insurance Regulatory and Development Authority of India).

---

## Dataset Description

- **Source:** Extracted from standard insurance policy documents, including Proposal Forms, Policy Schedules, and Product Benefit Tables.
- **Content:** 
  - Preamble and introductory clauses.
  - Definitions of key insurance terms (e.g., Accident, Insured Person, Proposer).
  - Policy terms, conditions, exclusions, and coverage details.
  - References to regulatory enactments and guidelines.

---

## Dataset Structure

The dataset is provided in JSON format with the following fields:

| Field          | Description                                                     |
|----------------|-----------------------------------------------------------------|
| `section`      | Policy section name (e.g., Preamble, Definitions)               |
| `clause_id`    | Unique identifier for each clause or definition                 |
| `term`        | Term or clause title (e.g., Accident)                            |
| `description`  | Full text description or definition                              |
| `references`   | Related regulatory references or cross-references (optional)    |
| `keywords`     | List of keywords extracted for search and indexing              |

Example entry:

```json
{
  "section": "Definitions",
  "clause_id": "def_001",
  "term": "Accident",
  "description": "Accident means a sudden, unforeseen and involuntary event caused by external, visible and violent means.",
  "references": ["IRDAI guidelines"],
  "keywords": ["accident", "unforeseen", "event", "external", "violent"]
}
