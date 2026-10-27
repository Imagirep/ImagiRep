"""Plain-text zero-shot gate prompts for GPT-6.1-sol on the ESTIMATED single-view setting (2026-10-09).

The gate sees exactly the `start_final` evidence (Start image + estimated FINAL VIEW with their captions) and the question
with options (trailing "Answer:" removed). Gate 1: call one extra estimated representation or answer directly
(NO -> the existing `start_final` answer). Gate 2: which representation (-> the existing `start_final_<m>` answer).
Descriptions state how each estimated representation is produced and what it cannot show (pipeline facts); no accuracy
is quoted. The top-down map is the re-rendered version whose final-facing arrow is clearly visible.
"""
SYSTEM = ('You help decide what evidence to use for a spatial reasoning question. '
          'You do not answer the question.')
QUESTION_HEADER = 'The question the reasoner must answer (do not answer it yourself):\n'
EST_NOTE = ('Note: the FINAL VIEW above was not photographed. A world model generated a 3D scene from the START image '
            'alone, and the FINAL VIEW was rendered in that generated scene after following the movement. It may be blurry '
            'or wrong, especially for things the START image does not show.')
ORDER = ('topdown', 'text', 'video', 'depth', 'surface_normals', 'panorama')

# d1: plain descriptions of each estimated representation (pipeline facts only).
DESC = {'d1': {
    'topdown': 'an overhead map of the generated scene: a grey, unlabeled point outline in which floor, walls and furniture '
               'look alike, with the numbered positions, the path, and an arrow at the final position showing the final '
               'facing direction. It does not show object names or colors.',
    'text': 'a written step-by-step description of the planned movement: distances walked and turns in degrees, ending '
            'with the final facing direction relative to the start. It contains no imagery.',
    'video': 'a first-person video through the generated scene that begins at the START view and follows the movement '
             'step by step; its last frame is exactly the FINAL VIEW.',
    'depth': 'a depth map of the generated final view: how far each visible surface is from the camera. It covers only '
             'what the final view shows.',
    'surface_normals': 'a surface normal map of the generated final view: which way each visible surface faces (floor, '
                       'walls, sides of objects). It covers only what the final view shows.',
    'panorama': 'a 360-degree image rendered at the final position from the world model\'s 3D scene. That scene is '
                'generated from the START image alone, so directions the START image does not show (usually the sides '
                'and behind) are filled in by the world model and may be missing, invented or misplaced.',
}}

# d2: pipeline facts from the asset audit.
# Top-down: the height filter keeps the floor, so open floor and obstacles look alike on the map. All representations
# come from the same generated scene, so none of them adds content that the generated scene lacks.
DESC['d2'] = dict(DESC['d1'])
DESC['d2']['topdown'] = DESC['d1']['topdown'] + (' Because the floor itself is drawn in the same grey as walls and '
                                                 'furniture, open floor and obstacles are hard to tell apart on it.')

# d3: the panorama text states the
# consequence of its generation for such questions: left / right / behind the final position are usually outside the
# START image, i.e. the invented part. Video text adds that it shows the turns along the path.
DESC['d3'] = dict(DESC['d2'])
DESC['d3']['panorama'] = DESC['d1']['panorama'] + (
    ' In particular, the directions a question asks about after the movement (to the left, right or behind the final '
    'position) usually lie outside the START image, so for those directions the panorama shows mostly invented content '
    'rather than the real room.')
DESC['d3']['video'] = DESC['d1']['video'] + ' It shows how the START view connects to the final position and how the view turns along the way.'

GATE1 = {
    'v2': ('Besides the images above, the reasoner can get ONE extra estimated representation of the generated scene, '
           'for example a top-down map, a written description of the movement, a video of the movement, a depth map, '
           'a surface normal map, or a 360-degree panorama. They are generated, not real, but they can help connect '
           'the START image to the final position.\n'
           'Say NO only if the START image and the FINAL VIEW together clearly show everything the question needs. '
           'Otherwise say YES.\n'
           'Reply with one short sentence giving your reason, then a final line that is exactly "DECISION: YES" or '
           '"DECISION: NO".'),
    'v3': ('Besides the images above, the reasoner can get ONE extra estimated representation of the generated scene, '
           'for example a top-down map, a written description of the movement, a video of the movement, a depth map, '
           'a surface normal map, or a 360-degree panorama. They are generated, not real, but they can help connect '
           'the START image to the final position.\n'
           'Say NO only if all of the following hold: every object or place the question names is clearly visible in '
           'the FINAL VIEW; the answer can be read directly from the FINAL VIEW; and the question does not depend on '
           'directions outside the FINAL VIEW (to the sides or behind) or on tracking how the START view relates to '
           'the final position. If any of these is uncertain, say YES.\n'
           'Reply with one short sentence giving your reason, then a final line that is exactly "DECISION: YES" or '
           '"DECISION: NO".'),
}
# v4: selective. All extra representations are rendered from
# the same generated scene as the FINAL VIEW, so they help only when the needed information is outside what the two
# images show (pipeline fact). YES only in that case.
GATE1['v4'] = ('Besides the images above, the reasoner can get ONE extra estimated representation, for example a top-down '
               'map, a written description of the movement, a video of the movement, a depth map, a surface normal map, '
               'or a 360-degree panorama. They are all rendered from the same generated scene as the FINAL VIEW, so they '
               'cannot add details that this scene got wrong; they only show other parts or other forms of it.\n'
               'Say YES only if the question needs something that neither the START image nor the FINAL VIEW shows, '
               'for example an object or direction outside both images, or where the path went. If the two images '
               'already contain what the question asks about, say NO.\n'
               'Reply with one short sentence giving your reason, then a final line that is exactly "DECISION: YES" or '
               '"DECISION: NO".')

GATE2 = {'v1': ('The reasoner will get ONE extra estimated representation of the generated scene. The options are:\n{options}\n'
                'Choose the one that most directly supplies the spatial information this question needs, taking into account '
                'what each option cannot show.\n'
                'Reply with one short sentence giving your reason, then a final line that is exactly "CHOICE: <name>", where '
                '<name> is one of: topdown, text, video, depth, surface_normals, panorama.')}


def gate1_text(v):
    return EST_NOTE + '\n\n' + GATE1[v]


def gate2_text(v, dv='d1'):
    return EST_NOTE + '\n\n' + GATE2[v].format(options='\n'.join(f'- {k}: {DESC[dv][k]}' for k in ORDER))
