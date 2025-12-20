#!/bin/bash

# Number of sequential jobs to submit
NUM_JOBS=5

# Ask user to paste their SLURM command
echo "Paste your SLURM command:"
read -r ORIGINAL_CMD

# Check if the command already contains a dependency flag
if [[ $ORIGINAL_CMD == *"--dependency"* ]]; then
  # Extract the existing dependency
  DEPENDENCY=$(echo $ORIGINAL_CMD | grep -o -- "--dependency=[^ ]*")
  # Remove the existing dependency to add our placeholder
  SBATCH_CMD=${ORIGINAL_CMD/$DEPENDENCY/DEPENDENCY_PLACEHOLDER}
else
  # Find the position after 'sbatch' to insert our dependency placeholder
  SBATCH_CMD=$(echo $ORIGINAL_CMD | sed 's/sbatch /sbatch DEPENDENCY_PLACEHOLDER /')
fi

# Submit the first job (with original dependency if it existed)
if [[ $ORIGINAL_CMD == *"--dependency"* ]]; then
  FIRST_CMD=${SBATCH_CMD/DEPENDENCY_PLACEHOLDER/$DEPENDENCY}
else
  # If no original dependency, the first job has no dependency
  FIRST_CMD=${SBATCH_CMD/DEPENDENCY_PLACEHOLDER/}
fi

job_id=$(eval "$FIRST_CMD" | grep -oP '(?<=Submitted batch job )\d+')
echo "Submitted job 1 with ID: $job_id"

# Submit the remaining jobs with dependencies on the previous job
for ((i=2; i<=NUM_JOBS; i++)); do
  next_cmd=${SBATCH_CMD/DEPENDENCY_PLACEHOLDER/--dependency=afterany:$job_id}
  job_id=$(eval "$next_cmd" | grep -oP '(?<=Submitted batch job )\d+')
  
  echo "Submitted job $i with ID: $job_id"
done
