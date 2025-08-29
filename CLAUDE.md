# Claude Instructions

## Project Overview
This is a nanoGPT implementation with muP (maximal update parametrization) support. The main focus is to test Mixture of Experts (MoE) with muP. The ultimate goal is to find hyperparameter transfer in MoE models.

## Development Notes
- Follow existing code style and conventions
- Do not make unnecessary changes to the codebase. When only asking for visualization of experiment results, do not modify how the experiment is run.
- If the request is not clear, ask for clarification.

## Important:
1: When I ask you to plan only, create and then write a plan file in txt format. In the plan file, include the following:
   (a) Section 1: Summarize the request in detailed technical terms, use math formula if needed.
   (b) Section 2: Write a detailed plan of execution steps that contains items to be completed in what order.
   (c) Section 3: For each step in Section 2, write the file name and functionality to be edited / created. For each file, write where and what (i.e. either change existing functionality or create new function).

2: When I ask you to review a plan, read the referred txt file following the above structure. Check if (a) the described steps are necessary and sufficient to complete the request and (b) whether or not the steps could trigger unwanted bugs or undesirable features. If one or more of the sections are not clear, ask for clarification.