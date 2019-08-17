**The assignment is not yet released for the Fall 2019 and might be subject to change.**

# Assignment 5 - Expectation Maximization

Expectation Maximization - Assignment 5 - CS6601 

<img src="images/k6_bird_color_24.png" width="400"/> <img src="images/pcd_clustered.gif" width="400"/> 

# Setup

Clone this repository:

`git clone https://github.gatech.edu/omscs6601/assignment_5.git`

The grading script runs with Python 3.6.7
(Do not use any version of Python 2.X)

#### Recommended - Use a virtualenv:

```bash
pip3 install virtualenv
virtualenv assignment_5
source assignment_5/bin/activate 
```
#### Requirements: 
Next you should install python packages listed in `requirements.txt`. You can install them using the command below:

`pip3 install -r requirements.txt`

#### Jupyter Notebook:
In order to complete this assignment, you will be using **jupyter notebook** instead of **.py** scripts as you did in earlier assignments. 

To open the Jupyter Notebook, navigate to your assignment folder, (activate your environment if you have/using one), and run `jupyter notebook`. 

Project description and all of the functions required to implement you will find in the `mixture_models.ipynb` file.

**ATTENTION:** You are free to add additional cells for debugging your implementation, however, please don't write any inline code in the cells with function declarations, only edit the section *inside* the function, which has comments like: `# TODO: finish this function`.

## Grading

The grade you receive for the assignment will be distributed as follows:

1. k-Means Clustering (19 points)
2. Gaussian Mixture Model (40 points)
3. Model Performance Improvements (20 points)
4. Bayesian Information Criterion (20 points)
5. Return your name (1 point)
6. Bonus (+5 points) (Bonus points are added to this assignment's grade and not to the overall grade.)


## Due Date
The assignment is due **March 31st, 2019 at 11:59PM UTC-12 (Anywhere on Earth time)**. The deliverable for this assignment is a completed mixture_models.ipynb file.

## Submission
The tests for the assignment are provided in `mixture_tests.py`, all of the tests are already embedded into the respective ipython notebook cells, so the will be running automaticagically when you run the cells with your code. Local tests are sufficient for verifying the correctness of your implementation, so Bonnie is only for submission purposes. The tests on Bonnie will be similar to the ones provided here, but the images and data being tested against, and the values for calculations will be different.

#### You will be allowed only 5 submissions on Bonnie. Make sure you test everything before submitting. The code will be allowed to run for not more than 1 hour per submission. In order for the code to run quickly, make sure to vectorize the code (more on this in the notebook itself).

### Since there are only 5 submissions allowed, your BEST submission will be used for the assignment grade.

To submit your assignment code to Bonnie run:

`python3 submit.py`

To submit your bonus code to Bonnie run:

`python3 submit_bonus.py`

As a backup, please also submit generated `mixture_models.ipynb` file to Canvas.

## Resources

1. Udacity lectures on Unsupervised Learning (Lesson 7)
2. The `gaussians.pdf`  in the `read/` folder will introduce you to multivariate normal distributions.
3. A youtube video by Alexander Ihler, on multivariate EM algorithm details:
[https://www.youtube.com/watch?v=qMTuMa86NzU](https://www.youtube.com/watch?v=qMTuMa86NzU)
4. The `em.pdf` chapter in the `read/` folder. This will be especially useful for Part 2 of the assignment.  
