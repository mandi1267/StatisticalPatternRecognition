Amanda Adkins
COMP136
Programming Project 2

This is the code written for http://www.cs.tufts.edu/comp/136/HW/pp2.pdf.

To execute:
- Run "python3 project2.py" on the homework.eecs.tufts.edu servers
- Note: All test files, along with __init__.py, proj2_common.py, proj2_task1.py,
  proj2_task2.py, and proj2_task3.py must be in the same directory as project2.py

File structure:
- proj2_common.py: Includes data structures and utility functions needed by all tasks
- proj2_task1.py: Includes evaluator for task1, and plotting utilities, since only task 1 involves plots
- proj2_task2.py: Includes evaluator for task2 and cross validation functionality
- proj3_task3.py: Includes evaluator for task3
- project2.py: Includes main function for instantiating and running all task
        evaluators, along with file reading and training/test set construction methods
