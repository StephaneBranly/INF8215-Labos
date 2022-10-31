include "globals.mzn";

int: n = 5;

array[1..n] of int: t_b = [2,2,3,2,1]; % top to bottom
array[1..n] of int: b_t = [3,2,1,3,4]; % bottom to top
array[1..n] of int: l_r = [4,1,3,2,3]; % left to right
array[1..n] of int: r_l = [1,4,2,2,2]; % right to left

array[1..n, 1..n] of var 1..n: x;      % hauteur des tours à chaque position

% Toutes les tours ont différentes tailles pour les lignes et pour les colonnes
constraint forall(i in 1..n)(alldifferent(j in 1..n)(x[i,j]));
constraint forall(j in 1..n)(alldifferent(i in 1..n)(x[i,j]));

% Nombre de tours visibles sur les différents axes
constraint forall(i in 1..n)(sum(j in 1..n)(max(x[i,1..j]) = x[i,j]) = l_r[i]); 
constraint forall(i in 1..n)(sum(j in 1..n)(max(x[i,j..n]) = x[i,j]) = r_l[i]); 

constraint forall(j in 1..n)(sum(i in 1..n)(max(x[1..i,j]) = x[i,j]) = t_b[j]); 
constraint forall(j in 1..n)(sum(i in 1..n)(max(x[i..n,j]) = x[i,j]) = b_t[j]); 

solve satisfy;

output 

["Configuration : \n"]

++

[ show(x[i,j]) ++ if j = n then "\n" else "," endif | i in 1..n, j in 1..n]
