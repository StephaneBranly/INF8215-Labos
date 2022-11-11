% Group Members :
%
% BRANLY Stéphane (MATRICULE 2232279)
% GUICHARD Amaury (MATRICULE 2227083)
%

include "globals.mzn";

enum PROJECT;
enum WORKER = { ARCHI, ING, ELEC };

int: P; % nombre de jours (et de projets à choisir)

array[PROJECT, PROJECT] of int: travel_costs;     % Couts de deplacements entre projets

set of PROJECT: project_with_precedences;         % Ensemble des projets ayant des contraintes de précédences
array[PROJECT, 1..3] of WORKER: precedences;      % Précédences (pour les projets n'ayant pas de contraintes de précédences, ignorez la rangée)

set of PROJECT: project_must_be_done;             % Ensemble des projets qui DOIVENT êtres faits

array[PROJECT] of int: revenu;                    % Revenu des projets

array[WORKER, 1..P] of var PROJECT: schedule;     % Pour chaque employé, pour chaque jour, on assigne un projet
array[WORKER, PROJECT] of var 0..P: day;          % Pour chaque employé, pour chaque projet, on détermine le jour 

% Profit (gain - dépenses) a maximiser
var int: profit;

% Seulement un worker par projet par jour
constraint forall(p in 1..P)(alldifferent(w in WORKER)(schedule[w,p]));

% Chaque worker travaille sur un projet different par jour
constraint forall(p in 2..P,w in WORKER)(schedule[w,p-1] != schedule[w,p]);

% Chaque worker travaille sur un projet qu'une seule fois
constraint forall(w in WORKER)(alldifferent(p in 1..P)(schedule[w,p]));

% Link entre schedule et day
constraint forall(p in 1..P, w in WORKER)(day[w, schedule[w,p]] == p);

% Projets devant etre finis
array[PROJECT] of var bool: projet_fini;
constraint forall(p in PROJECT)(if count(schedule, p) == 3 then projet_fini[p] == true else projet_fini[p] == false endif);

% Projets ayant des précédences
constraint forall(p in project_with_precedences, w in 2..3)(day[precedences[p,w-1],p] < day[precedences[p,w],p]);

var int: depenses;
var int: revenus;

% Calcul des dépenses
constraint sum(p in 2..P, w in WORKER)(travel_costs[schedule[w,p-1], schedule[w,p]]) = depenses;

% Calcul des revenus
constraint sum(p in PROJECT where projet_fini[p] == true)(revenu[p]) = revenus;

% Projets devant etre faits
constraint forall(p in project_must_be_done)(projet_fini[p] == true);

% Calcul du profit
constraint profit = revenus - depenses;

% Maximisation du profit
solve maximize profit;
output

["Horaire : \n"] 
++
[if p = 1 then show(w) ++ ":" else "" endif ++ show(schedule[w,p]) ++ if p = P then "\n" else "," endif | w in WORKER, p in 1..P]
++
["-----------\n"]
++
["Profit : " ++ show(profit)]
