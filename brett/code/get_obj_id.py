shapes = ["cube", "sphere", "cylinder"]
materials = ["metal", "rubber"]
colors = ["gray", "red", "blue", "green", "brown", "cyan", "purple", "yellow"]

def get_id(the_object):
    color = the_object['color']
    material = the_object['material']
    shape = the_object['shape']
    
    c_id = colors.index(color)
    m_id = materials.index(material)
    s_id = shapes.index(shape)
    
    obj_id = s_id * 16 + m_id * 8 + c_id + 1
    
    return obj_id