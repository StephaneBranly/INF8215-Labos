include "globals.mzn";

var int: nombre;
array[1..4] of var 1..9: chiffres;

constraint alldifferent(chiffres);
constraint chiffres[4] mod 2 == 0;
constraint chiffres[1]>chiffres[2];
constraint chiffres[4]>chiffres[3];
constraint chiffres[2]>chiffres[4];
constraint sum(chiffres)>15;

solve minimize chiffres[2]*chiffres[3]*chiffres[4];

nombre = chiffres[4] + chiffres[3]*10 + chiffres[2]*100 + chiffres[1]*1000;

output [
  show(nombre)
];


