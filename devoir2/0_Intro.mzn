% Group Members :
%
% BRANLY Stéphane (MATRICULE 2232279)
% GUICHARD Amaury (MATRICULE 2227083)
%

include "globals.mzn";

var int: nombre;
array[1..4] of var 1..9: chiffres;

% Quatre chiffres sont differents
constraint alldifferent(chiffres);

% Nombre pair
constraint chiffres[4] mod 2 == 0;

% Chiffre des milliers supérieur à celui des centaines
constraint chiffres[1]>chiffres[2];

% Chiffre des dizaines inférieur à celui des unités
constraint chiffres[4]>chiffres[3];

% Chiffre des centaines supérieur à celui des unités
constraint chiffres[2]>chiffres[4];

% Somme des chiffres supérieure à 15
constraint sum(chiffres)>15;

% Minimisation du produit des trois derniers chiffres
solve minimize chiffres[2]*chiffres[3]*chiffres[4];

nombre = chiffres[4] + chiffres[3]*10 + chiffres[2]*100 + chiffres[1]*1000;

output [
  show(nombre)
];


