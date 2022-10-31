include "globals.mzn";

% Nombre de segments horizontaux et verticaux
int: h; % nb de rows
int: v; % nb de cols

% La demande par segment horizontal et vertical
array[1..h] of int: demand_per_h;
array[1..v] of int: demand_per_v;

% Le coût du permis et de construction de maison associé à chaque carré
array[1..h, 1..v] of int: cost_per_house;
array[1..h, 1..v] of int: cost_per_permit;

% Le nombre max de maison qui peut être construit par carré
int: m;

% Le nombre de maisons à construire par carré
% Moins de m maisons par carré et au moins 0 maisons par carré
array[1..h, 1..v] of var 0..m: x;


% Nombre de maisons minimales par segment (ligne h et colonne v)
constraint forall(i in 1..h)(sum(j in 1..v)(x[i,j])>=demand_per_h[i]);
constraint forall(j in 1..v)(sum(i in 1..h)(x[i,j])>=demand_per_v[j]);



% Le coût total
var int: tot_cost;

% Contrainte sur le coup total des constructions
constraint sum(i in 1..h, j in 1..v where x[i,j] > 0)(x[i,j]*cost_per_house[i,j] + cost_per_permit[i,j]) = tot_cost;

solve minimize tot_cost;

output
  
  ["Configuration : \n"]  
  
  ++
  
  [
    show(x[i,j]) ++ if j = v then "\n" else "," endif 
    | i in 1..h, j in 1..v
  ]
  
  ++
  
  ["----------------\n"]
  
  ++
  
  [
  "Cost : " ++ show(tot_cost)
  ]
  