dataset_info = dict(
    dataset_name='spnv2',
    keypoint_info={
        0:
        dict(name='shoulder1', id=0, color=[51, 153, 255], type='', swap=''),
        1:
        dict(
            name='shoulder2',
            id=1,
            color=[51, 153, 255],
            type='',
            swap=''),
        2:
        dict(
            name='shoulder3',
            id=2,
            color=[51, 153, 255],
            type='',
            swap=''),
        3:
        dict(
            name='shoulder4',
            id=3,
            color=[51, 153, 255],
            type='',
            swap=''),
        4:
        dict(
            name='ankle1',
            id=4,
            color=[255, 128, 0],
            type='',
            swap=''),
        5:
        dict(
            name='ankle2',
            id=5,
            color=[255, 128, 0],
            type='',
            swap=''),
        6:
        dict(
            name='ankle3',
            id=6,
            color=[255, 128, 0],
            type='',
            swap=''),
        7:
        dict(
            name='ankle4',
            id=7,
            color=[255, 128, 0],
            type='',
            swap=''),
        8:
        dict(
            name='line1',
            id=8,
            color=[0, 255, 0],
            type='',
            swap=''),
        9:
        dict(
            name='line2',
            id=9,
            color=[0, 255, 0],
            type='',
            swap=''),
        10:
        dict(
            name='line3',
            id=10,
            color=[0, 255, 0],
            type='',
            swap='')
    },
    skeleton_info={
        0:
        dict(link=('shoulder1', 'shoulder2'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('shoulder2', 'shoulder3'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('shoulder3', 'shoulder4'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('shoulder4', 'shoulder1'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('shoulder1', 'ankle1'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('shoulder2', 'ankle2'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('shoulder3', 'ankle3'), id=6, color=[51, 153, 255]),
        7:
        dict(link=('shoulder4', 'ankle4'),id=7, color=[51, 153, 255]),
        8:
        dict(link=('ankle1', 'ankle2'), id=8, color=[0, 255, 0]),
        9:
        dict(
            link=('ankle2', 'ankle3'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('ankle3', 'ankle4'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('ankle4', 'ankle1'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('shoulder1', 'line1'), id=12, color=[51, 153, 255]),
        13:
        dict(link=('shoulder3', 'line2'), id=13, color=[51, 153, 255]),
        14:
        dict(link=('shoulder4', 'line3'), id=14, color=[51, 153, 255])
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.
    ],
    sigmas=[
        0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05
    ])
