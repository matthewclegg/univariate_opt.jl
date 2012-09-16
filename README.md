univariate_opt.jl
=================

Univariate optimization and root-finding code for Julia

Usage
=====

<code>
min(F::Function, x0::Float64, x1::Float64)
max(F::Function, x0::Float64, x1::Float64)
argmin(F::Function, x0::Float64, x1::Float64)
argmax(F::Function, x0::Float64, x1::Float64)
zero(F::Function, x0::Float64, x1::Float64)
inv(F::Function, x0::Float64, x1::Float64)
polynomial_roots(a::Vector{Float64})
</code>

Details
=======

The function min(F, x0, x1) (respectively max(F, x0, x1)) searches for
a local minimum (maximum) of F in the closed interval [x0,x1].  It
returns the value F(x) that is found to minimize (maximize) F.

The algorithm first searches for a bracket that is known to contain a
local minimum (maximum).  From there, a modified version of Brent's
method is used to precisely locate the minimum (maximum).  Convergence
is guaranteed to be linear and is often much more rapid.  For further
details, see section 10.3 of "Numerical Recipes".

The function argmin(F, x0, x1) is identical to min(F, x0, x1) except
that the value x that minimizes F is returned, rather than returning
F(x).  The function argmax(F, x0, x1) is analogous.

The function zero(F, x0, x1) searches for a value x within the closed
interval [x0,x1] such that F(x) = 0.  Unlike the R function uniroot(),
it is not a requirement that F(x0) and F(x1) have opposite signs.
Rather, the algorithm initially searches for an interval [a,b] such
F(a) and F(b) have opposite signs.  Once this is found, a modified
version of Ridder's method with bisection steps is used to precisely
locate the zero.  Convergence is guaranteed to be at least linear, and
for well-behaved functions, super-linear convergence is obtained.  For
further details, see section 9.2 of "Numerical Recipes".

The function inv(F, x0, x1) constructs the inverse of F in the
interval [x0,x1].  In other words, a function G:Float64 -> Float64 is
returned such that G(F(x)) = x for all x in [x0,x1].  If F is strictly
monotone and continuous, then G also will be strictly monotone and
continuous.  If these conditions are not satisfied, then all bets are
off.

All of the preceding functions accept an optional fourth parameter,
tolerance::Float64, which specifies the tolerance to be used in
assessing the objective function.

The function polynomial_roots(a::Vector{Float64}) accepts as input a
vector of length n representing the coefficients of the polynomial

  a[1] + a[2] x + a[3] x^2 + ... + a[n] x^(n-1)

As output, it produces a pair (roots::Vector{Float64},
mult::Vector{Float64}) representing the real roots and their
multiplicities, e.g., roots[i] is a real root of multiplicity
multi[i].  This routine was provided as an illustration of the
zero-finding procedure, and no claims are made about its optimality.
When performance is an issue or when complex roots are needed, the
user might wish to consult the literature, starting with section 9.5
of "Numerical Recipes".

A few simple performance tests were conducted to assess the
performance of this code vis a vis the corresponding R routines.  In
the following, v is a vector of length 3,001, and w is a vector of
length 30,001.  These tests were conducted on a Mac Pro with 2.66 GHz
Xeon processors.

<pre>
                                                              Julia        R
 10000 reps of A:  min(x->x^2-1, -5., 0.)                     0.479 sec    0.681 sec
 10000 reps of B:  min(x->x^2-1, 1., 10.)                     0.477 sec    0.722 sec
 10000 reps of C:  min(x->^2-1, 1., 10.)                      0.159 sec    0.277 sec
 10000 reps of D:  zero(x->x^2-1, -5., 2.)                    0.107 sec    0.464 sec
  1000 reps of E:  min(x->sum((v - x).^2), -10., 20.)         0.391 sec    0.513 sec
   100 reps of F:  zero(x->sum(x - w.^2), -100., 200          0.581 sec    0.528 sec
</pre>

See the benchmark() routine for further details.

Examples
========

<pre>
julia> min(x->(x-2)*(x+3), -10., 10.)
-6.25

julia> argmin(x->(x-2)*(x+3), -10., 10.)
-0.5

julia> zero(x->(x-2)*(x+3), -10., 10.)
-2.999999999984872

julia> max(x->sin(x), 0., pi)
1.0

julia> argmax(x->sin(x), 0., pi)
1.5707963267948966

julia> zero(x->cos(x)-x, 0., 2*pi)
0.7390851331629317

julia> atan2 = inv(tan, -1.57, 1.57)
#<function>

julia> 4.0*atan2(1.)
3.1415926535918897

julia> atan2(1.)-atan(1.)  # Compare our arctan with the supplied version
5.241362899255364e-13

julia> polynomial_roots([-6., 1., 1.])  # x^2+x-6 = (x-2)*(x+3)
([-3.0, 2.0],[1, 1])

julia> H10=[-30240., 0., 302400., 0., -403200., 0., 161280., 0., -23040., 0., 1024.]
11-element Float64 Array:
...

julia> @time polynomial_roots(H10)
elapsed time: 0.055845022201538086 seconds
([-3.43616, -2.53273, -1.75668, -1.03661, -0.342901, 0.342901, 1.03661, 1.75668, 2.53273, 3.43616],[1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
</pre>

See also
========

optim.jl by John Myles White.  This is a package for multivariate
optimization.  In the tests that I have conducted, I have found this
implementation of the Nelder-Mead method to be quite competitive with
the R implementation.  https://github.com/johnmyleswhite/optim.jl

glm.jl by Douglas Bates.  This is a package for fitting generalized
linear models.  https://github.com/dmbates/glm.jl

References
==========

William H. Press, Saul A. Teukolsky, William T. Vetterling and Brian P. Flannery.  2007. "Numerical Recipes:  That Art of Scientific Computing, 3rd Edition" Cambridge: Cambridge University Press.

Author
======

Matthew Clegg
matthewcleggphd@gmail.com

Comments and suggestions are of course welcome.
