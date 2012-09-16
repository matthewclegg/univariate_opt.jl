module univariate_opt

# Routines for performing univariate optimization.
# Copyright (C) 2012 by Matthew Clegg
# Author's email: matthewcleggphd@gmail.com
# All comments and suggestions are welcome.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
# 
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# FURTHERMORE, I AM NOT A NUMERICAL ANALYST (ALTHOUGH, LIKE DOROTHY 
# HAYDEN TRUSCOTT, I DO HAVE TWO DEGREES IN MATH).

# This module contains various routines whose purpose is to assist
# in finding zeros and extrema of univariate functions.
#
# Let F:Float64 -> Float64 be a real-valued function, and let x0::Float64
# and x1::Float64 be real values.  The following functions are provided:
# 
# min(F, x0, x1) -- Searches for a local minimum of F within the interval [x0,x1],
#   and returns the minimum that is found.
# argmin(F, x0, x1) -- Searches for a local minimum of F within the interval [x0,x1],
#   and returns the value x::Float64 that gives rise to that minimum.
#   E.g., F(argmin(F, x0, x1)) == min(F, x0, x1)
# max(F, x0, x1) -- Searches for a local maximum of F within the interval [x0, x1],
#   and returns the maximum that is found.
# argmax(F, x0, x1) -- Searches for a local maximum within the interval [x0, x1],
#   and returns the value x::Float64 that gives rise to that maximum.
# zero(F, x0, x1) -- Searches for and returns a value x::Float64 within [x0,x1] 
#   such that F(x)=0
# inv(F, x0, x1) -- Returns a function G:[F(x0),F(x1)] -> [x0,x1] such that
#   F(G(x)) == x
#
# All of the above functions accept an optional fourth argument tol::Float64, which
# is the tolerance to be accepted in approximating the result.  If tol is omitted,
# then sqrt(eps(Float64)) is used, and if tol is given, it should be at least as large
# as this value.
#
# In the event that a search is unsuccessful, an exception is thrown.
#
# Note that there is a naming collision with min and max in the base module reduce.jl.
# In reduce.jl, min(F, x0, x1) = min(F(x0), x1), and similarly for max.
#
# This code is largely derived from algorithms given in "Numerical Recipes:
# The Art of Scientific Computing, 3rd Edition" by William Press, Saul Teukolsky, 
# William Vetterling and Brian Flannery.  Cambridge University Press, 1992.
#
# Examples
#
# julia> min(x->(x-2)*(x+3), -10., 10.)
# -6.25
#
# julia> argmin(x->(x-2)*(x+3), -10., 10.)
# -0.5
#
# julia> zero(x->(x-2)*(x+3), -10., 10.)
# -2.999999999984872
#
# julia> max(x->sin(x), 0., pi)
# 1.0
#
# julia> argmax(x->sin(x), 0., pi)
# 1.5707963267948966
#
# julia> zero(x->cos(x)-x, 0., 2*pi)
# 0.7390851331629317
#
# julia> atan2 = inv(tan, -1.57, 1.57)
# #<function>
#
# julia> 4.0*atan2(1.)
# 3.1415926535918897
#
# julia> atan2(1.)-atan(1.)  # Compare our arctan with the supplied version
# 5.241362899255364e-13
#
# julia> polynomial_roots([-6., 1., 1.])  # x^2+x-6 = (x-2)*(x+3)
# ([-3.0, 2.0],[1, 1])
#
# julia> H10=[-30240., 0., 302400., 0., -403200., 0., 161280., 0., -23040., 0., 1024.]
# 11-element Float64 Array:
# ...
#
# julia> @time polynomial_roots(H10)
# elapsed time: 0.055845022201538086 seconds
# ([-3.43616, -2.53273, -1.75668, -1.03661, -0.342901, 0.342901, 1.03661, 1.75668, 2.53273, 3.43616],[1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

import Base.*
export min, argmin, max, argmax, zero, inv,
  find_zero_bracket, find_maximum_bracket, 
  golden_section_search, golden_section_search_with_brent_steps,
  findzero_using_bisection, findzero_using_bisection_and_ridder,
  test, benchmark, polynomial_roots

# Uncomment the body of the following macro if you would like to see debug information
macro debug(expr)
#  expr
end

# Maximum precision of approximation functions
const min_eps = sqrt(eps(Float64))::Float64    # Is ::Float64 necessary or helpful?

# The following internal type is defined to simplify the implementation of
# of find_...bracket.
type Bracket
  a::Float64   # left end-point of the interval
  b::Float64   # right end-point of the interval
  fa::Float64  # F(a)
  fb::Float64  # F(b)
end

function find_maximum_bracket(
  F::Function,      # The function for which a bracket is sought
  p0::Float64,      # The left endpoint of the interval to search
  p1::Float64,      # The right endpoint of the interval to search
  fp0::Float64,     # F(p0)
  fp1::Float64,     # F(p1)
  maxprobe::Int64)  # Maximum number of evaluations of the objective function
  # Searches for three points p0<=x1<x2<x3<=p1 that bracket a maximum of the function F.
  # E.g., such that f(x1) < f(x2) and f(x3) < f(x2).  These three points can then be
  # used in a more refined search algorithm to locate a maximum value of F within [p0,p1].
  # If no bracket can be found after maxprobe function evaluations, throws an exception.  
  # Otherwise, returns a sextuple (x1, x2, x3, F(x1), F(x2), F(x3))

  bracket_list = [Bracket(p0,p1,fp0,fp1)]
  for i = 3:maxprobe  # Why start at 3? Because F has been called twice already
    iv = shift(bracket_list)
    c = 0.5 * (iv.a + iv.b)
    fc = F(c)::Float64
    if (fc > iv.fa) && (fc > iv.fb)
      return (iv.a, c, iv.b, iv.fa, fc, iv.fb)
    end
    @debug println("bracketing $([iv.a,c,iv.b]) -> $([iv.fa, fc, iv.fb])")
    push(bracket_list, Bracket(iv.a, c, iv.fa, fc))
    push(bracket_list, Bracket(c, iv.b, fc, iv.fb))
  end
  
  error ("find_maximum_bracket could not find a bracket")
end
  
function find_maximum_bracket(
  F::Function,      # The function for which a bracket is sought
  p0::Float64,      # The left endpoint of the interval to search
  p1::Float64,      # The right endpoint of the interval to search
  maxprobe::Int64)  # Maximum number of evaluations of the objective function
  
  find_maximum_bracket(F, p0, p1, F(p0), F(p1), maxprobe)
end

function find_maximum_bracket(
  F::Function,        # The function for which a bracket is sought
  p0::Float64,        # The left endpoint of the initial interval to search
  p1::Float64,        # The right endpoint of the initial interval to search
  fp0::Float64,       # F(p0)
  fp1::Float64,       # F(p1)
  extend_left::Bool,  # true if the search should extend to the left of p0
  extend_right::Bool, # true if the search should extend to the right of p1
  maxprobe::Int64)    # Maximum number of times the objective function can be evaluated
  # Searches for three points p0<=x1<x2<x3<=p1 that bracket a maximum of the function F.
  # E.g., such that f(x1) < f(x2) and f(x3) < f(x2).  These three points can then be
  # used in a more refined search algorithm to locate a maximum value of F within [x1,x3].
  # If extend_left is true, then the search interval is extended to the left as the search progresses.
  # Similarly, if extend_right is true, then the search interval is extended to the right
  # as the search progresses.  If successful, returns a sextuple (x1, x2, x3, F(x1), F(x2), F(x3)) 
  # such that x1 < x2 < x3, F(x2) > F(x1) and F(x2) > F(x3).  If no bracket can be found after
  # maxprobe function evaluations, throws an exception.

  bracket_list = [Bracket(p0,p1,fp0,fp1)]
  left_bound = p0
  right_bound = p1
  for i = 3:maxprobe  # Why start at 3? Because F has been called twice already
    iv = shift(bracket_list)
    c = 0.5 * (iv.a + iv.b)
    fc = F(c)::Float64
    if (fc > iv.fa) && (fc > iv.fb)
      return (iv.a, c, iv.b, iv.fa, fc, iv.fb)
    end
    @debug println("bracketing $([iv.a,c,iv.b]) -> $([iv.fa, fc, iv.fb])")
    push(bracket_list, Bracket(iv.a, c, iv.fa, fc))
    push(bracket_list, Bracket(c, iv.b, fc, iv.fb))
    if extend_left && (iv.a == left_bound)
      left_bound = left_bound - 2 * (iv.b - iv.a)
      push(bracket_list, Bracket(left_bound, iv.a, F(left_bound)::Float64, iv.fa))
    end
    if extend_right && (iv.b == right_bound)
      right_bound = right_bound + 2 * (iv.b - iv.a)
      push(bracket_list, Bracket(iv.b, right_bound, iv.fb, F(right_bound)::Float64))
    end
  end
  
  error ("find_maximum_bracket could not find a bracket")
end

function find_maximum_bracket(
  F::Function,        # The function for which a bracket is sought
  p0::Float64,        # The left endpoint of the initial interval to search
  p1::Float64,        # The right endpoint of the initial interval to search
  extend_left::Bool,  # true if the search should extend to the left of p0
  extend_right::Bool, # true if the search should extend to the right of p1
  maxprobe::Int64)    # Maximum number of times the objective function can be evaluated
  
  find_maximum_bracket(F, p0, p1, F(p0), F(p1), extend_left, extend_right, maxprobe)
end

function find_zero_bracket(
  F::Function,       # The function for which a bracket is sought
  p0::Float64,        # The left endpoint of the interval to search
  p1::Float64,        # The right endpoint of the interval to search
  fp0::Float64,       # F(p0)
  fp1::Float64,       # F(p1)
  maxprobe::Int64)    # Maximum number of times the objective function can be evaluated
  # Searches for two points p0<=x0<x1<=p1 that bracket a zero of the function F,
  # which is assumed to be continuous.  E.g., such that either (i) f(x0) <= 0
  # and f(x1) > 0, or (ii) f(x0) > 0 and f(x1) <= 0.  Returns the quad (x0,x1,F(x0),F(x1)).
  # If no bracket can be found after maxprobe function evaluations, then throws
  # an exception.

  bracket_list = [Bracket(p0,p1,fp0,fp1)]
  nprobes = 2
  while length(bracket_list) > 0 
    iv = shift(bracket_list)
    if iv.fa * iv.fb <= 0.
      return (iv.a, iv.b, iv.fa, iv.fb)
    end
    if nprobes < maxprobe
      nprobes += 1
      c = 0.5 * (iv.a + iv.b)
      fc = F(c)::Float64
      @debug println("bracketing $([iv.a, c, iv.fa, fc])")
      push(bracket_list, Bracket(iv.a, c, iv.fa, fc))
      @debug println("bracketing $([c, iv.b, fc, iv.fb])")
      push(bracket_list, Bracket(c, iv.b, fc, iv.fb))
    end
  end
  
  error ("find_zero_bracket could not find a bracket")
end
 
function find_zero_bracket(
  F::Function,       # The function for which a bracket is sought
  p0::Float64,        # The left endpoint of the interval to search
  p1::Float64,        # The right endpoint of the interval to search
  maxprobe::Int64)    # Maximum number of times the objective function can be evaluated
  
  find_zero_bracket(F, p0, p1, F(p0), F(p1), maxprobe)
end 

function golden_section_search (
  F::Function,        # Function whose maximum is sought, Float64 -> Float64
  p1::Float64,        # Left endpoint of interval in which to search
  p2::Float64,        # Initial test point
  p3::Float64,        # Right endpoint of interval in which to search
  fp2::Float64,       # F(p2)
  tolerance::Float64) # Precision required in final answer
  # Searches for a maximum of F in the interval [p1,p3] using the
  # golden section search of Kiefer.  On input, p1, p2 and p3 must satisfy
  # the following conditions:
  #     p1 < p2 < p3
  #    f(p1) < f(p2)
  #    f(p3) < f(p2)
  # The search terminates when the change in the value of the
  # objective function is no more than tolerance.
  #
  # Kiefer, J. (1953), "Sequential minimax search for a maximum,"
  # Proceedings of the American Mathematical Society 4 (3): 502-506.
  #
  # See also http://en.wikipedia.org/wiki/Golden_section_search
    
  (x1,x2,x3) = (p1, p2, p3)
  fx2 = fp2
  
  resphi = 2.0 - (1. + sqrt(5.))/2.
  iter = 0
  fx2prev = fx2 - tolerance - 1
  while (fx2 - fx2prev > tolerance) && (x3 - x1 > tolerance) && (iter < 100)
    iter += 1
    xwidth = x3 - x1
    fx2save = fx2
    if x2 - x1 > x3 - x2
      x4 = x2 - resphi*(x2 - x1)
      fx4 = F(x4)::Float64
      if fx4 > fx2
        (x2,x3,fx2) = (x4,x2,fx4)
      else
        x1 = x4
      end
    else
      x4 = x2 + resphi*(x3 - x2)
      fx4 = F(x4)::Float64
      if fx4 > fx2
        (x1, x2, fx2) = (x2, x4, fx4)
      else
        x3 = x4
      end
    end
    if fx2 != fx2save
      fx2prev = fx2save
    end
    @debug print("$iter $([x1,x2,x3]) -> $fx2\n")
  end
  
  (x2, fx2)
end

function golden_section_search (
  F::Function,        # Function whose maximum is sought, Float64 -> Float64
  p1::Float64,        # Left endpoint of interval in which to search
  p2::Float64,        # Initial test point
  p3::Float64,        # Right endpoint of interval in which to search
  tolerance::Float64) # Precision required in final answer
  
  golden_section_search(F, p1, p2, p3, F(p2), tolerance)
end

function golden_section_search_with_brent_steps (
  F::Function,        # Function whose maximum is sought, Float64 -> Float64
  p1::Float64,        # Left endpoint of interval in which to search
  p2::Float64,        # Initial test point
  p3::Float64,        # Right endpoint of interval in which to search
  fp1::Float64,       # F(p1)
  fp2::Float64,       # F(p2)
  fp3::Float64,       # F(p3)
  tolerance::Float64) # Precision required in final answer
  # Searches for a maximum of F in the interval [p1,p3] using a
  # combination of golden section steps and Brent steps.  In a Brent step,
  # a parabola is fitted to the triple (p1,p2,p3), and the next test point
  # is chosen to be the extreme point of the parabola, if it is within
  # the interval [p1, p3].  See section 10.3 of "Numerical Recipes"
  # for a description of Brent's method.
  #
  # On input, p1, p2 and p3 must satisfy the following conditions:
  #     p1 < p2 < p3
  #    f(p1) < f(p2)
  #    f(p3) < f(p2)
  #
  # The search terminates when an interval [x1,x2,x3] is found such that
  # |F(x2) - F(x1)| < tolerance and |F(x2) - F(x3)| < tolerance.
  #
  # This code should be compared to Brent_fmin in the R package,
  # which seems to be a nearly verbatim implementation of the code as given
  # in "Numerical Recipes" (which itself is probably a verbatim copy of
  # Brent's original Algol code, but I have not verified this).
  #
  # In a few simple tests that I tried, this code outperformed
  # the R optimize() function.
  #
  # R.P. Brent (1973), "Algorithms for Minimization Without Derivatives,"
  # Chapter 4. Prentice-Hall, Englewood Cliffs, NJ.
  #
  # See also http://en.wikipedia.org/wiki/Brent%27s_method
    
  (x1, x2, x3) = (p1, p2, p3)
#  (fx1, fx2, fx3) = (F(x1)::Float64, F(x2)::Float64, F(x3)::Float64)
  (fx1, fx2, fx3) = (fp1, fp2, fp3)
  
  function update(x1::Float64, x2::Float64, x3::Float64, 
    fx1::Float64, fx2::Float64, fx3::Float64, x4::Float64)
    # On input [x1, x2, x3] is a bracketing triple and
    # [fx1, fx2, fx3] are the values of F at these points.
    # The point x4 is a test point between x1 and x3.  Calculates
    # f(x4) and returns a smaller bracketing sextuple
    #   (x1', x2', x3', fx1', fx2', fx3')
    # satisfying x1 <= x1' < x2' < x3' <= x3
    # with F(x2') > F(x1') and F(x2') > F(x3')
    fx4 = F(x4)::Float64
    if x4 < x2
      if fx4 > fx2
        (x2,x3,fx2,fx3) = (x4,x2,fx4,fx2)
      else
        (x1,fx1) = (x4, fx4)
      end
    else
      if fx4 > fx2
        (x1, x2, fx1, fx2) = (x2, x4, fx2, fx4)
      else
        (x3, fx3) = (x4, fx4)
      end
    end
    (x1, x2, x3, fx1, fx2, fx3)
  end
  
  phi = (1. + sqrt(5.))/2.
  resphi = 2.0 - phi
  phim1 = phi - 1.

  iter = 0
  w0 = 0.  # The proportion by which the search range was reduced 2 iterations ago
  w1 = 0.  # The proportion by which the search range was last reduced
  while (iter < 100) && ((fx2 - fx1 > tolerance) || (fx2 - fx3 > tolerance))
    iter += 1
    golden_section_step_needed = true
    xwidth = x3 - x1
    
    # Try to do a Brent step
    bnum = (x2-x1)^2*(fx2-fx3)-(x2-x3)^2*(fx2-fx1)
    bden = (x2-x1)*(fx2-fx3)-(x2-x3)*(fx2-fx1)
    x4 = x2 - 0.5 * bnum/bden
    xmin_eps = min_eps * x2 + tolerance/3
    # Avoid evaluating the function at points that are too close together.
    if abs(x4 - x2) < xmin_eps
      x4 = (x2 - x1 > x3 - x2)? (x2 - xmin_eps): (x2 + xmin_eps)
    end
    
    if (x4 - x1 > xmin_eps) && (x3 - x4 > xmin_eps)
      (x1, x2, x3, fx1, fx2, fx3) = update(x1, x2, x3, fx1, fx2, fx3, x4)
      (w0, w1) = (w1, (x3 - x1) / xwidth)
      # The following test imposes the condition that in order to avoid
      # a golden section step, the last two successive Brent steps must reduce
      # the width of the bracketing interval by at least as much as one 
      # golden section step.  This optimization was suggested in "Numerical
      # Recipes".
      if w0 * w1 < phim1
        golden_section_step_needed = false
      end
      iter += 1
      @debug print("$iter Brent $([x1,x2,x3]) -> $([fx1, fx2, fx3])\n")
    end    

    # Check if a golden section step is needed and do it
    if golden_section_step_needed
      if x2 - x1 > x3 - x2
        x4 = x2 - resphi*(x2 - x1)
      else
        x4 = x2 + resphi*(x3 - x2)
      end
      (x1, x2, x3, fx1, fx2, fx3) = update(x1, x2, x3, fx1, fx2, fx3, x4)
      (w0, w1) = (w1, (x3 - x1) / xwidth)
      
      @debug print("$iter GSS   $([x1,x2,x3]) -> $([fx1, fx2, fx3])\n")
    end
  end
  
  (x2, fx2)
end

function golden_section_search_with_brent_steps (
  F::Function,        # Function whose maximum is sought, Float64 -> Float64
  p1::Float64,        # Left endpoint of interval in which to search
  p2::Float64,        # Initial test point
  p3::Float64,        # Right endpoint of interval in which to search
  tolerance::Float64) # Precision required in final answer
  
  golden_section_search_with_brent_steps(F, p1, p2, p3, F(p1), F(p2), F(p3), tolerance)
end

function argmax (
  F::Function,        # Function whose maximum is sought, Float64 -> Float64
  x0::Float64,        # Left endpoint of interval in which to search
  x1::Float64,        # Right endpoint of interval in which to search
  tol::Float64)       # Precision required in final answer
  # Searches for a point x::Float64 in the interval [x0,x1] such that
  # F(x) is a local maximum of F.  Returns x.
  
  fx0 = F(x0)
  fx1 = F(x1)

  local b
  try
    b = find_maximum_bracket(F, x0, x1, fx0, fx1, 50)
  catch
    return (fx0 > fx1)? x0: x1
  end
  
  (x, fx) = golden_section_search_with_brent_steps(F, b..., tol)

  # If we've reached this point, then we've definitely found a local max already.
  # Nonetheless, one of the endpoints could offer an even larger value,
  # so why not check?
  if (fx >= fx0) && (fx >= fx1)
    xmax = x
  elseif (fx0 >= fx) && (fx0 >= fx1)
    xmax = x0
  else
    xmax = x1
  end
  
  xmax
end

argmax(F::Function, x0::Float64, x1::Float64) = argmax(F, x0, x1, min_eps)

function max (
  F::Function,        # Function whose maximum is sought, Float64 -> Float64
  x0::Float64,        # Left endpoint of interval in which to search
  x1::Float64,        # Right endpoint of interval in which to search
  tol::Float64)       # Precision required in final answer
  # Searches for a point x::Float64 in the interval [x0,x1] such that
  # F(x) is a local maximum of F.  Returns F(x)
 
  fx0 = F(x0)
  fx1 = F(x1)

  local b
  try
    b = find_maximum_bracket(F, x0, x1, fx0, fx1, 50)
  catch
    return (fx0 > fx1)? fx0: fx1
  end
  
  (x, fx) = golden_section_search_with_brent_steps(F, b..., tol)
  
  # If we've reached this point, then we've definitely found a local max already.
  # Nonetheless, one of the endpoints could offer an even larger value,
  # so why not check?
  if (fx >= fx0) && (fx >= fx1)
    fxmax = fx
  elseif (fx0 >= fx) && (fx0 >= fx1)
    fxmax = fx0
  else
    fxmax = fx1
  end
  
  fxmax
end

max(F::Function, x0::Float64, x1::Float64) = max(F, x0, x1, min_eps)

function argmin (
  F::Function,        # Function whose maximum is sought, Float64 -> Float64
  x0::Float64,        # Left endpoint of interval in which to search
  x1::Float64,        # Right endpoint of interval in which to search
  tol::Float64)       # Precision required in final answer
  # Searches for a point x::Float64 in the interval [x0,x1] such that
  # F(x) is a local minimum of F.  Returns x. 

  argmax(x -> -F(x), x0, x1, tol)
end

argmin(F::Function, x0::Float64, x1::Float64) = argmin(F, x0, x1, min_eps)

function min (
  F::Function,        # Function whose minimum is sought, Float64 -> Float64
  x0::Float64,        # Left endpoint of interval in which to search
  x1::Float64,        # Right endpoint of interval in which to search
  tol::Float64)       # Precision required in final answer
  # Searches for a point x::Float64 in the interval [x0,x1] such that
  # F(x) is a local minimum of F.  Returns F(x)

  -max(x -> -F(x), x0, x1, tol)
end

min(F::Function, x0::Float64, x1::Float64) = min(F, x0, x1, min_eps)

function findzero_using_bisection (
  F::Function,        # The function for which a zero is sought
  x0::Float64,        # The left endpoint of an interval containing the zero
  x1::Float64,        # The right endpoint of an interval containing the zero
  fx0::Float64,       # F(x0)
  fx1::Float64,       # F(x1)
  tolerance::Float64) # Tolerance to be accepted in approximating the zero
  # Uses the bisection method to find a value x such that F(x) = 0.
  # On input, it is assumed that F is continuous and that F(x0) and F(x1)
  # have opposite signs.  Returns a value x such that abs(F(x)) <= tolerance.
  
  if fx0 * fx1 > 0
    error ("Endpoints of findzero_using_bisection must have opposite signs")
  end
  
  if abs(fx0) <= tolerance
    return x0
  elseif abs(fx1) <= tolerance
    return x1
  end
  
  xmid = (x0 + x1) * 0.5
  fxmid = F(xmid)
  iter = 0
  while (abs(fxmid) > tolerance) && (iter < 100)
    iter += 1
    if sign(fxmid) == sign(fx0)
      (x0, fx0) = (xmid, fxmid)
    else
      (x1, fx1) = (xmid, fxmid)
    end
    @debug println("$([x0, x1, fx0, fx1])")
    xmid = (x0 + x1) * 0.5
    fxmid = F(xmid)
  end
  xmid
end

function findzero_using_bisection_and_ridder (
  F::Function,        # The function for which a zero is sought
  x0::Float64,        # The left endpoint of an interval containing the zero
  x1::Float64,        # The right endpoint of an interval containing the zero
  fx0::Float64,       # F(x0)
  fx1::Float64,       # F(x1)
  tolerance::Float64) # Tolerance to be accepted in approximating the zero
  # Uses Ridder's method to find a value x such that F(x) = 0.
  # On input, it is assumed that F is continuous and that F(x0) and F(x1)
  # have opposite signs.  Returns a value x such that abs(F(x)) <= tolerance.
  #
  # This implementation uses a combination of bisection steps and steps using
  # Ridders' Method.  Ridder's method is described in section 9.2.1 of 
  # "Numerical Recipes".  This implementation is slightly different
  # than the one given.  It is guaranteed that each iteration will shrink 
  # the interval by at least 1/2.

  if fx0 * fx1 > 0
    error ("Endpoints of findzero_using_bisection_and_ridder must have opposite signs")
  end
  
  if abs(fx0) <= tolerance
    return x0
  elseif abs(fx1) <= tolerance
    return x1
  end
  
  (x3, fx3) = (x1, fx1)
  iter = 0
  while (abs(fx3) > tolerance) && (iter < 100)
    iter += 1
    x2 = (x0 + x1) * 0.5
    fx2 = F(x2)
    if abs(fx2) < tolerance
      return x2
    end

    s = sqrt(fx2^2 - fx0*fx1)
    if s == 0.
      return x2
    end
    x3 = x2 + (x2  - x0) * sign(fx0 - fx1) * fx2 / s
    fx3 = F(x3)
    @debug print("$iter $([x0, x1, x2, x3]) -> $([fx0, fx1, fx2, fx3]) ")
    
    if sign(fx2) == sign(fx0)
      @debug print("B0 ")
      (x0, fx0) = (x2, fx2)
    else
      @debug print("B1 ")
      (x1, fx1) = (x2, fx2)
    end
    if (sign(fx3) == sign(fx0)) && (x0 < x3)
      @debug print("R0 ")
      (x0, fx0) = (x3, fx3)
    elseif (sign(fx3) == sign(fx1)) && (x3 < x1)
      @debug print("R1 ")
      (x1, fx1) = (x3, fx3)
    end
    @debug println()
    x2 = (x0 + x1) * 0.5
  end
  x3
end

fubar(F, x0, x1, fx0, fx1, tol) = findzero_using_bisection_and_ridder(F, x0, x1, fx0, fx1, tol)

function zero (
  F::Function,        # Function whose zero is sought, Float64 -> Float64
  x0::Float64,        # Left endpoint of interval in which to search
  x1::Float64,        # Right endpoint of interval in which to search
  tol::Float64)       # Precision required in final answer
  # Searches for a point x::Float64 in the interval [x0,x1] such that
  # F(x) = 0.  Returns x.
 
  fx0 = F(x0)
  fx1 = F(x1)
  try
    (x0, x1, fx0, fx1) = find_zero_bracket(F, x0, x1, fx0, fx1, 50)
  catch
    error ("zero was unable to find a bracket containing a zero")
  end
  
  @debug println("initial bracket $([x0, x1, fx0, fx1])")
  fubar(F, x0, x1, fx0, fx1, tol)
end

zero(F::Function, x0::Float64, x1::Float64) = zero(F, x0, x1, min_eps)

function inv (
  F::Function,        # Function whose inverse is sought, Float64 -> Float64
  x0::Float64,        # Left endpoint of interval where F is to be inverted
  x1::Float64,        # Right endpoint of interval
  tol::Float64)       # Precision required in final answer
  # Returns a function G:Float64 -> Float64 such that F(G(x)) = x
 
  fx0 = F(x0)
  fx1 = F(x1)
  (fxmin, fxmax) = (min(fx0, fx1), max(fx0, fx1))
  
  function invF(a::Float64, F, x0, x1, fx0, fx1, fxmin, fxmax)
    if (a < fxmin) || (fx1 < fxmax) 
      error ("Call to inverse function with value outside of domain")
    end
    
    local p0, p1, fp0, fp1
    try
      (p0, p1, fp0, fp1) = find_zero_bracket(x -> F(x) - a, x0, x1, fx0-a, fx1-a, 50)
    catch
      error ("zero was unable to find a bracket containing a zero")
    end
  
    @debug println("initial bracket $([x0, x1, fx0, fx1])")
    fubar(x -> F(x) - a, p0, p1, fp0, fp1, tol)
  end
  
  return x -> invF(x, F, x0, x1, fx0, fx1, fxmin, fxmax)
end

inv(F::Function, x0::Float64, x1::Float64) = inv(F, x0, x1, min_eps)

macro assert_exception(ex)
  quote
    ok = true
    try
      $ex
      ok = false
    catch
      nothing
    end
    if !ok
      error("assert_exception failed: ", $string(ex))
    end
  end
end

approx_eq(X,Y,tol) = abs(X - Y) <= tol
approx_eq(X,Y) = approx_eq(X,Y,0.00001)
  
function each_approx_eq(X,Y,tol)
  if length(X) != length(Y)
    return false
  end
  for i in 1:length(X)
    if abs(X[i] - Y[i]) > tol
      return false
    end
  end
  return true
end

each_approx_eq(X,Y) = each_approx_eq(X,Y,0.00001)

function test()
  # A collection of unit tests.  Returns true if all tests pass, otherwise
  # reports an error and generates an exception.
  
  # find_maximum_bracket
  @assert_exception find_maximum_bracket(x -> -x^2, -1., 1., 1)
  @assert each_approx_eq(find_maximum_bracket(x -> -x^2, -1., 1., 3), (-1., 0., 1., -1., 0., -1.))
  @assert_exception find_maximum_bracket(x -> x^2, -1., 1., 1)
  @assert_exception find_maximum_bracket(x -> x^2, -1., 1., 1000)
  @assert each_approx_eq(find_maximum_bracket(x->-x^2, 1., 2., true, false, 100), (-1., 0., 1., -1., 0., -1.))
  @assert_exception find_maximum_bracket(x->-x^2, 1., 2., false, true, 100)
  @assert each_approx_eq(find_maximum_bracket(x->-x^2, 1., 2., true, true, 100), (-1., 0., 1., -1., 0., -1.))
  @assert each_approx_eq(find_maximum_bracket(x->-x^2, -2., -1., false, true, 100), (-1., 0., 1., -1., 0., -1.))
  
  # find_zero_bracket
  @assert find_zero_bracket(x->x, -1., 1., 10) == (-1., 1., -1., 1.)
  @assert find_zero_bracket(x->x, 0., 1., 10) == (0., 1., 0., 1.)
  @assert find_zero_bracket(x->x, -1., 0., 10) == (-1., 0., -1., 0.)
  @assert_exception find_zero_bracket(x->x, 1., 2., 10)
  @assert find_zero_bracket(x->-x, -1., 1., 10) == (-1., 1., 1., -1.)
  @assert_exception find_zero_bracket(x->(x-1)*(x+1), -10., 50., 10)
  @assert find_zero_bracket(x->(x-1)*(x+1), -10., 50., 20) == (-2.5, -0.625, 5.25, -0.609375)
  
  # golden_section_search
  @assert each_approx_eq(golden_section_search(x->sin(x), 0., 1., 2.0, 0.), (1.57079,1.0))
  @assert each_approx_eq(golden_section_search(x->sin(x), 0., 1., 2.0, 0.1), (1.61803,0.99888))
  @assert each_approx_eq(golden_section_search(x->-(x-1./3.)^2, 0., 0.5, 1., 0.000000001), (1./3., 0.))
  
  # golden_section_search_with_brent_steps
  gsswbs(F, x0, x1, x2, t) = golden_section_search_with_brent_steps(F, x0, x1, x2, t)
  @assert each_approx_eq(gsswbs(x->sin(x), 0., 1., 2.0, 0.), (1.57079,1.0))
  @assert each_approx_eq(gsswbs(x->-(x-1./3.)^2, 0., 0.5, 1., 0.0000001), (1./3., 0.))
  
  f2(x) = (x-1)*(x-2)
  f3(x) = (x-1)*(x-2)*(x-3)
  f4(x) = (x-1)*(x-2)*(x-3)*(x-4)
  
  # argmax
  @assert approx_eq(argmax(x->x, 0., 1.), 1.)
  @assert approx_eq(argmax(x->-x, 0., 1.), 0.)
  @assert approx_eq(argmax(x->-f2(x), -5., 5.), 1.5)
  @assert approx_eq(argmax(x->-f2(x), 1., 5.), 1.5)
  @assert approx_eq(argmax(x->-f3(x), 0., 5.), 0.)
  @assert approx_eq(argmax(x->-f3(x), 1., 5.), 2.5773502)
  @assert approx_eq(argmax(x->f3(x), 0., 5.), 5.)
  @assert approx_eq(argmax(x->-f4(x), 0., 2.5), 1.38196)
  @assert approx_eq(argmax(x->-f4(x), 2.5, 5.), 3.61803)
  @assert approx_eq(argmax(x->sin(x), 0., pi), pi/2.)
  
  # max
  @assert approx_eq(max(x->x, 0., 1.), 1.)
  @assert approx_eq(max(x->-x, 0., 1.), 0.)
  @assert approx_eq(max(x->2., 0., 1.), 2.)
  @assert approx_eq(max(x->-f2(x), -5., 5.), 0.25)
  @assert approx_eq(max(x->-f2(x), 1., 5.), 0.25)
  @assert approx_eq(max(x->-f3(x), 0., 5.), 6.)
  @assert approx_eq(max(x->-f3(x), 1., 5.), 0.3849002)
  @assert approx_eq(max(x->f3(x), 0., 5.), 24.)
  @assert approx_eq(max(x->-f4(x), 0., 2.5), 1.)
  @assert approx_eq(max(x->-f4(x), 2.5, 5.), 1.)
  @assert approx_eq(max(x->sin(x), 0., pi), 1.)

  # argmin
  @assert approx_eq(argmin(x->x, 0., 1.), 0.)
  @assert approx_eq(argmin(x->-x, 0., 1.), 1.)
  @assert approx_eq(argmin(x->f2(x), -5., 5.), 1.5)
  @assert approx_eq(argmin(x->f2(x), 1., 5.), 1.5)
  @assert approx_eq(argmin(x->f3(x), 0., 5.), 0.)
  @assert approx_eq(argmin(x->f3(x), 1., 5.), 2.5773502)
  @assert approx_eq(argmin(x->-f3(x), 0., 5.), 5.)
  @assert approx_eq(argmin(x->f4(x), 0., 2.5), 1.38196)
  @assert approx_eq(argmin(x->f4(x), 2.5, 5.), 3.61803)
  @assert approx_eq(argmin(x->cos(x), 0., 3. * pi/2.), pi)

  # min
  @assert approx_eq(min(x->x, 0., 1.), 0.)
  @assert approx_eq(min(x->-x, 0., 1.), -1.)
  @assert approx_eq(min(x->2., 0., 1.), 2.)
  @assert approx_eq(min(x->f2(x), -5., 5.), -0.25)
  @assert approx_eq(min(x->f2(x), 1., 5.), -0.25)
  @assert approx_eq(min(x->f3(x), 0., 5.), -6.)
  @assert approx_eq(min(x->f3(x), 1., 5.), -0.3849002)
  @assert approx_eq(min(x->-f3(x), 0., 5.), -24.)
  @assert approx_eq(min(x->f4(x), 0., 2.5), -1.)
  @assert approx_eq(min(x->f4(x), 2.5, 5.), -1.)
  @assert approx_eq(min(x->cos(x), 0., 3. * pi/2.), -1.)

  # findzero_using_bisection
  fub(F, x0, x1, fx0, fx1, tol) = findzero_using_bisection(F, x0, x1, fx0, fx1, tol)
  @assert approx_eq(fub(x->0., 0., 1., 0., 0., 0.0000001), 0.)
  @assert approx_eq(fub(x->x, -1., 1., -1., 1., 0.0000001), 0.)
  @assert_exception fub(x->x, 1., 2., 1., 2., 0.0000001)
  @assert approx_eq(fub(x->x^2-1., 0., 3., -1., 8., 0.0000001), 1.)
  @assert approx_eq(fub(x->x^2-1., -2., 0., 3., -1., 0.0000001), -1.)
  
  # findzero_using_bisection_and_ridder
  @assert approx_eq(fubar(x->0., 0., 1., 0., 0., 0.0000001), 0.)
  @assert approx_eq(fubar(x->x, -1., 1., -1., 1., 0.0000001), 0.)
  @assert_exception fubar(x->x, 1., 2., 1., 2., 0.0000001)
  @assert approx_eq(fubar(x->x^2-1., 0., 3., -1., 8., 0.0000001), 1.)
  @assert approx_eq(fubar(x->x^2-1., -2., 0., 3., -1., 0.0000001), -1.)
  
  # zero 
  @assert approx_eq(zero(x->0., 0., 1.), 0.)
  @assert approx_eq(zero(x->x, -1., 1.), 0.)
  @assert_exception zero(x->x, 1., 2.)
  @assert approx_eq(zero(x->x^2-1., 0., 3.), 1.)
  @assert approx_eq(zero(x->x^2-1., -2., 0.), -1.)

  # inv
  g1 = inv(x->x^2, 0., 1.)
  @assert_exception g1(-1.)
  @assert g1(0.) == 0.
  @assert approx_eq(g1(0.5), sqrt(0.5))
  @assert g1(1.) == 1.
  @assert_exception g1(2.)
  
  atan2 = inv(tan, -1.5, 1.5)
  atan_err = max(map(x->atan2(x)-atan(x), (-100:100)*0.01))
  @assert atan_err < sqrt(eps(Float64))
  
  # polynomial_roots
  function pcheck(p, r, m)
    (rx,mx) = polynomial_roots(p)
    @assert each_approx_eq(r, rx)
    @assert each_approx_eq(m, mx)
  end
  pcheck(Array(Float64, 0), [], [])
  pcheck([1.], [], [])
  pcheck([1., 1.], [-1.], [1])
  pcheck([2., 2.], [-1.], [1])
  pcheck([1., 0., 1.], [], [])
  pcheck([-1., 0., 1.], [-1., 1.], [1, 1])
  pcheck([-1., 0., 1., 0.], [-1., 1.], [1, 1])
  pcheck([1., 2., 1.], [-1.], [2])
  pcheck([1., 3., 3., 1.], [-1.], [3])
  pcheck([-6.,11.,-6.,1.], [1.0, 2.0, 3.0], [1, 1, 1])
  pcheck([-1,0.,0.,1.], [1.0], [1])
  pcheck([24.,14.,-13.,-2.,1.], [-3.,-1.,2.,4.], [1,1,1,1])
  pcheck([-16.,4.,12.,-7.,1.], [-1.,2.,4.], [1,2,1])
  pcheck([2., 0., 3., 0., 1.], [], [])
  pcheck([-120.,274.,-225.,85.,-15.,1.], [1.,2.,3.,4.,5.], [1,1,1,1,1])
  
#  pcheck([-1., 0., 0., 1.]
  println("all tests passed in univariate_opt.jl")
  true 
end

macro benchtime(label, nreps, func)
  quote
    @printf("%6d reps of %-45s", $nreps, $label)
    gc()
    local t0 = time()
    for i in 1:$nreps
      $func
    end
    local elapsed = time() - t0
    @printf("%8.4f sec\n", elapsed)
  end
end

function benchmark()
  # A collection of performance benchmarks.
  
  # The first benchmark tests the performance of the optimization functions
  # on the simple polynomial f(x) = x^2 - 1.  Since this function is so easy
  # to compute, the benchmark is really measuring the overhead imposed by Julia
  # and by the way in the optimization functions have been coded.
  
  # R: system.time(for(i in 1:10000) optimize(function(x) x^2-1, c(-5.,0.)))
  @benchtime "A:  min(x^2-1, -5., 0.)"  10000 min(x->x^2-1, -5., 0.)
  
  # R: system.time(for(i in 1:10000) optimize(function(x) x^2-1, c(1.,10.)))
  @benchtime "B:  min(x^2-1, 1., 10.)"  10000 min(x->x^2-1, 1., 10.)

  # R: system.time(for(i in 1:10000) optimize(function(x) x^2-1, c(-1.,10.)))
  @benchtime "C:  min(x^2-1, 1., 10.)"  10000 min(x->x^2-1, -1., 10.)
  
  # R: system.time(for(i in 1:10000) uniroot(function(x) x^2-1, c(-5.,0.)))
  @benchtime "D:  zero(x^2-1, -5., 2.)" 10000 zero(x->x^2-1, -5., 0.)
  
  # The second benchmark tests the performance of finding the value that
  # minimizes a sum of squares, where the number of terms in the sum is large.
  # In this benchmark, the cost of function evaluation overwhelms the bookkeeping
  # overhead, and so this benchmark is a measure of the extent to which our
  # minimization function keeps the number of calls to the objective function down.
  v=(-1000:2000)*0.01
  f(x::Float64) = sum((v - x).^2)

  function unrolled_sum(x::Vector{Float64})
    z = 0.0
    for i in 1:length(v)
      z += v[i]
    end
    z
  end
  
  f2(x::Float64) = unrolled_sum((v-x).^2)
  
  function f3(x::Float64)
    z = 0.0
    for i in 1:length(v)
      z += (v[i]-x)^2
    end
    z
  end
  
  # R: v = (-1000:2000)*0.01
  #    system.time(for(i in 1:1000) optimize(function(x) sum((v - x)^2), c(-10.,20.)))
  @benchtime "E:  min(sum((v - x).^2), -10., 20.)" 1000 min(f, -10., 20.)
  @benchtime "E': min(unrolled_sum((v - x).^2), -10., 20.)" 100 min(f2, -10., 20.)
  
  # The third benchmark is a very backwards way of calculating the mean of a
  # set of numbers.  The purpose of this benchmark is to evaluate the efficiency
  # of the zero finding procedure in terms of the number of calls to the
  # objective function.
  
  # R:  w = (-10000:20000)*0.001
  #     system.time(for(i in 1:100) uniroot(function(x) sum(x-w^2), c(-100., 200.)))
  w=(-10000:20000)*0.001
  g(x) = sum(x - w.^2)
  @benchtime "F:  zero(sum(x - w^2), -100., 200.)" 100 zero(g, -100., 200.)
  
  #  Following are the times that were recorded on 9/16/2012 on a Mac Pro
  #  with 2.66 GHz Xeon processors:
  #
  #         Julia       R
  #  A      0.489   0.681
  #  B      0.478   0.722
  #  C      0.156   0.277
  #  D      0.107   0.464
  #  E      0.391   0.513
  #  F      0.581   0.528  
  #
  #  Only test F proved to be slower in Julia that R.  I investigated the 
  #  number of evaluations of the objective function in this test, and I
  #  found that my implementation only evaluates it four times, whereas the
  #  R uniroot function evaluates it five times.  
  #
  #  One possible explanation for the slower time for (F) is that
  #  the sum() operation is significantly slower in Julia than in R.
  #  Compare the following:
  #
  #  Julia:
  #    v = float64(1:10000)*0.1
  #    @time for i=1:10000 sum(v) end
  #    elapsed time: 0.41809606552124023 seconds
  #
  #  R:
  #    v=(1:10000)*0.1
  #    system.time(for(i in 1:10000) sum(v))
  #    user  system elapsed 
  #    0.204   0.000   0.204 
end

function polynomial_roots (a::Vector{Float64}) 
  # Searches for the real roots of the polynomial 
  #    f(x) = a[1] + a[2] x + a[3] x^2 + ... + a[n] x^(n-1)
  # Returns a pair (roots::Vector{Float64}, mult::Vector{Int64}) where
  # roots[i] is a real root of f of multiplicity mult[i].
  #
  # This algorithm is a toy algorithm that was written for the purpose of testing
  # the zero finding procedure.  For more sophisticated algorithms, see section 9.5
  # of Numerical Recipes.
  #
  # This algorithm works recursively as follows.  If f is of degree two (or less),
  # we use the quadratic formula.  Otherwise, suppose we know the real roots of the
  # derivative f' of f.  Let these roots be d[1] < d[2] < ... < d[m].  In the open
  # interval (d[i], d[i+1]), the sign of f' is always positive or always negative.
  # Consequently, in this region, f is either strictly monotone increasing or strictly
  # monotone decreasing.  Therefore, f can have at most one root in this region.  
  # If f[d[i]] has the same sign as f[d[i+1]], then there are no roots in this region. 
  # If f[d[i]] has a different sign than f[d[i+1]], then there is exactly one root
  # of multiplicity one, and we can use one of our above zero finding techniques to
  # identify it.  In addition, there may be a root at d[i] or d[i+1] (but not both).
  # These will be multiple roots.
  #
  # The running time of this algorithm is O(n^2 b), where b is the amount of time
  # required to find a root that has been bracketed.

  if length(a) <= 1
    return (Array(Float64,0), Array(Int64,0))
  elseif a[end] == 0.
    n = length(a)
    while (n > 0) && (a[n] == 0.)
      n -= 1
    end
    return polynomial_roots(a[1:n])
  elseif length(a) == 2
    # f is linear
    return ([-a[1]/a[2]], [1])
  elseif length(a) == 3
    # f is a quadratic.  Check the discriminant of the quadratic formula
    # to determine how many real roots f has.
    disc = a[2]^2 - 4 * a[1] * a[3]
    if abs(disc) < min_eps
      return ([-a[2]/(2*a[3])], [2])
    elseif disc < 0
      return (Array(Float64,0), Array(Int64,0))
    else
      return ([(-a[2]-sqrt(disc))/(2*a[3]), (-a[2]+sqrt(disc))/(2*a[3])], [1,1])
    end
  end
  
  # Get the zeros of the derivative of f
  d = [(i - 1) * a[i] for i=2:length(a)]
  (dzeros,dmult) = polynomial_roots(d)
  
  # Normalize a so that the coefficient of x^(n-1) is 1.
  anorm = [a[i] / a[end] for i=1:length(a)]
  
  n = length(anorm) - 1       # degree of polynomial
  fzeros = Array(Float64,0)   # Roots that have been found so far
  fmult = Array(Int64,0)      # Multiplicities of those roots.
  f(x) = sum([anorm[i] * x^(i-1) for i = 1:length(anorm)])

  # xend is a value that is large enough such that for all z > xend, 
  # sign(f(-z)) == sign(f(-xend)) and sign(f(z)) == sign(f(xend)).  
  # f cannot have any zeros larger than xend or smaller than -xend.
  #
  # The value xend is chosen so as to satisfy
  #   xend^(i-1-n) * |anorm[i]| <= 1/(n+1)
  # for each i.  From this, it can easily be worked out that f(z) > 0 
  # for all z >= xend.
  xend = max([(1.0/((n+1) * abs(anorm[i])))^(1.0/(i-1-n)) for i = 1:n])
  
  dzeros = [-xend, dzeros..., xend]
  dmult = [0, dmult..., 0]
  
  for i = 1:(length(dzeros)-1)
    if abs(f(dzeros[i])) <= min_eps
      push(fzeros, dzeros[i])
      push(fmult, dmult[i]+1)
    elseif abs(f(dzeros[i+1])) <= min_eps
      continue
    elseif f(dzeros[i])*f(dzeros[i+1]) < 0
      z = zero(f, dzeros[i], dzeros[i+1])
      push(fzeros, z)
      push(fmult, 1)
    end
  end
      
  (fzeros, fmult)
end

end # module
