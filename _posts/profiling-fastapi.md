---
title: Profiling FastAPI with PyInstrument
author: Matt Badger
date: 2022-11-22
categories: python, profiling
---

## Introduction

We use [FastAPI]() and [Mangum]() to build AWS API Gateways on top of Lambda functions
for almost all of our microservices at Branch. Python not being the _fastest_
programming language on the planet, being mindful of the potential environmental impact,
and making sure that end users get the results they need from our APIs as quickly as
possible, we have a nice standard way of measuring performance with
[PyInstrument]()

## PyInstrument

PyInstrument is a great little tool for profiling your Python code, and comes with
