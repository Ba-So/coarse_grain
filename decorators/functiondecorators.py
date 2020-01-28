#!/usr/bin/env python
# coding=utf-8
import debugdecorators as dd

def requires(requirements):
  def decorator(func):
    def function(*args, **kwargs):
      setattr(func, 'needs', requirements)
      return func(*args, **kwargs)
    return function
  return decorator
