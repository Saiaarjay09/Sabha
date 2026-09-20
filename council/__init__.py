"""Hiring Council — a panel of deliberately biased AI assessors that reads a
CV against a specific job and argues about it.

The design goal is the opposite of keyword matching. Nothing in here compares
strings from a job description to strings in a CV. Instead a job description is
first decomposed into atomic *requirements*, each carrying an explicit
statement of what evidence would prove it and which adjacent experience counts
as a legitimate proxy. Only then is the CV read, as an indexed body of
evidence, and each requirement is judged against it — including the question
keyword matching can never answer: "this person has not done exactly this, but
can they do it?"
"""

__version__ = "1.0.0"
