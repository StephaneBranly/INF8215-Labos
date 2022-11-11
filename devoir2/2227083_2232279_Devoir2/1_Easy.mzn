% Group Members :
%
% BRANLY Stéphane (MATRICULE 2232279)
% GUICHARD Amaury (MATRICULE 2227083)
%

include "globals.mzn";

enum ATTRACTION;

% Il faut sélectionner entre l et u attractions
int: l;
int: u;

% Attrait touristique total minimum
int: m;

% Attrait touristique par paire d'attraction
array[ATTRACTION, ATTRACTION] of int: pair_attraction;

% Coût de construction par attraction
array[ATTRACTION] of int: cost;

% Ensemble d'attractions à construire
var set of ATTRACTION: x;

% Coût total
var int: tot_cost;

% Attrait touristique supérieur à m
constraint sum(i in x, j in x where i > j)(pair_attraction[i,j]) > m;

% Sélection entre l et u attractions
constraint card(x) <= u;
constraint card(x) >= l;

% Calcul du cout total
constraint tot_cost = sum(i in x)(cost[i]);

% Minimisation du cout total
solve minimize tot_cost;


output [
  "Attractions : ", show(x), "\n",
  "Cost : ", show(tot_cost)
];