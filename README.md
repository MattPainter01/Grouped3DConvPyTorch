# Motivation
Grouped convolutions for most cases should be faster than group-1 convolutions since there is less parameters and less operations. In pytorch 2D grouped convolutions are faster than their group-1 counterparts but for 3D convolutions this is not the case. 

The table demonstrates this for 1000 forward-backward iterations of convolutions with spatial size 3 and N input and output channels.

|    Groups     | 2D            | 3D            |
| ------------- | ------------- | ------------- |
|      1        | Content Cell  | Content Cell  |
|      N        | Content Cell  | Content Cell  |

