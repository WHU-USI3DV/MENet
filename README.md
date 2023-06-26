# MENet 

:triangular_flag_on_post:	The whole codebase is coming soon. 

:triangular_flag_on_post: MENet++ is on the road.

[![demo_nus](imgs/demo_nus.gif)](https://youtu.be/YEzCHtUVj8M "Click to youtube")

## Performance

Performance without CBGS:
|                      Method                       | Modality |  mAP  |  NDS  |                                                Weights                                                 |
| :-----------------------------------------------: | :------: | :---: | :---: | :---------------------------------------------------------------------------------------------------: |
|      [CenterPoint](./configs/centerpoint.py)      |    L     | 52.7  | 61.2  | [Google Drive](https://drive.google.com/file/d/13nqSh3hfjdYKecchp_0JutKJK_Y-0hKE/view?usp=share_link) |
|            [MENet](./configs/menet.py)            |   L+M    | 56.9  | 63.4  | [Google Drive](https://drive.google.com/file/d/10Fex_w3k8KASrJ0x3zNqZTVrZmU40wgF/view?usp=share_link) |
|    [SECOND](./configs/hv_second_secfpn_nus.py)    |    L     | 35.7  | 51.3  | [Google Drive](https://drive.google.com/file/d/1l0MDs8h88ymvDE023SXZRyOg4xe8igw4/view?usp=share_link) |
| [ME SECOND](./configs/me_hv_second_secfpn_nus.py) |   L+M    | 43.1  | 55.5  | [Google Drive](https://drive.google.com/file/d/1KpYIGs248xoMo6oYiUEtDSe4d0MH1UzO/view?usp=share_link) |

Performance with CBGS:
|                Method            | Modality |  mAP  |  NDS  |         Weights      |
| :------------------------------------: | :------: | :---: | :---: | :-----------: |
|      CenterPoint   |   L    | 56.2 | 64.7  | TODO |
|      MENet   |   L+M    | 56.7  | 65.5  | TODO |
|    SECOND  |    L     | 47.6  | 59.2  | TODO |
| ME SECOND |   L+M    | 50.9  | 61.4  | TODO |
