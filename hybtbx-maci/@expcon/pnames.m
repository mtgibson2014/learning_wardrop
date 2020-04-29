function [Props,AsgnVals] = pnames(EXPCON)
%PNAMES  All public properties and their assignable values
%
%   [PROPS,ASGNVALS] = PNAMES(EXPCON)  returns the list PROPS of
%   public properties of the object EXPCON (a cell vector), as well as 
%   the assignable values ASGNVALS for these properties (a cell vector
%   of strings).  PROPS contains the true case-sensitive property names.
%
%   See also  GET, SET.

%   (C) 2003 by A. Bemporad


% EXPCON public properties

% Remember to be consistent with EXPCON.M in case of empty @EXPCON objects
Props = {'H';'K';'F';'G';'i1';'i2';'nr';'thmin';'thmax';...
        'nu';'npar';'ts';'model';'nx';'ny';'norm';'cost';'Observer';'info'};

% Also return assignable values if needed
if nargout>1,
   AsgnVals = {
         'q-by-npar array       (hyperplanes, lhs)';...
         'q-by-1 array          (hyperplanes, rhs)';...
         '(nr*nu)-by-npar array (linear gains)';...
         '(nr*nu)-by-1 array    (constant terms)';...
         '1-by-nr array         (j-th region: H(i1(j):i2(j),:)*th<=K(i1(j):i2(j)) )';...
         '1-by-nr array';...
         'scalar                (number of regions)';...
         'npar-by-1 array       (lower bounds on parameters)';...
         'npar-by-1 array       (upper bounds on parameters)';...
         'scalar                (number of inputs)';...
         'scalar                (number of parameters)';...
         'scalar                (sampling time)';...
%         'nr-by-1 array         (Chebychev radii of polyhedra)';...
%         'scalar                (tolerance for considering a polyhedron flat)';...
%         'scalar {0,1}          (problem is constrained/unconstrained)';...
%         'nr-by-3 array         (RGB colors for plotting 2D regions)';...
%          'scalar                (flag: from @LINCON object)';...
%          'scalar                (flag: from @HYBCON object)';...
%          'scalar                (flag: from @MPC object)';...
%          'char                  (name of implicit controller that originated the solution)';...
         'model object          (LTI or MLD)';...
         'scalar                (number of states)';...
         'scalar                (number of outputs)';...
%          'scalar                (flag: 1=tracking, 0=regulation)';...
%          'scalar                (flag: 1=constrained, 0=unconstrained)';...
%          'struct                (reference signals -- only hybrid)';...
%          'struct                (sizes of reference signals -- only hybrid)';...
%          'struct                (indices of fixed reference signals -- only hybrid)';...
%          'struct                (values of fixed reference signals -- only hybrid)';...
%          'struct                (indices of fixed measured disturbances -- only MPC)';...
%          'struct                (values of fixed measured disturbances -- only MPC)';...
         'scalar                (norm used, either 2 or Inf)';...
         'struct                (cost matrices -- only hybrid, 2-norm)';...
         'char array or struct  (observer information)';...
         'struct                (additional informations)';...
};
end

% end expcon/pnames.m