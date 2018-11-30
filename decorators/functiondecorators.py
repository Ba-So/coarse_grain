#!/usr/bin/env python
# coding=utf-8

def requires(requirements):
  def decorator(func):
    def function(*args, **kwargs):
      setattr(func, 'needs', requirements)
      return func(*args, **kwargs)
    return function
  return decorator
