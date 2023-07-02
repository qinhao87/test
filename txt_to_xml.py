import xml.dom.minidom
def to_xml(txt_dir,xml_dir,num):
    fp=open(txt_dir)
    lines=fp.readlines()

    doc=xml.dom.minidom.Document()
    annotation=doc.createElement("annotation")
    doc.appendChild(annotation)

    nodefloder=doc.createElement('floder')
    nodefloder.appendChild(doc.createTextNode('15'))

    filename=doc.createElement('filename')
    filename.appendChild(doc.createTextNode(str(num)))

    source=doc.createElement('source')
    database=doc.createElement('database')
    database.appendChild(doc.createTextNode('Unkown'))
    source.appendChild(database)

    size=doc.createElement('size')
    width=doc.createElement('width')
    width.appendChild(doc.createTextNode('720'))
    height=doc.createElement('height')
    height.appendChild(doc.createTextNode('720'))
    depth=doc.createElement('depth')
    depth.appendChild(doc.createTextNode('3'))
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)

    segmented=doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))

    annotation.appendChild(nodefloder)
    annotation.appendChild(filename)
    annotation.appendChild(source)
    annotation.appendChild(size)
    annotation.appendChild(segmented)

    for line in lines:
        objectlist=line.rstrip().split()

        object=doc.createElement('object')

        name=doc.createElement('name')
        name.appendChild(doc.createTextNode(str(objectlist[0])))

        pose=doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))

        truncated=doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('0'))

        difficult=doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))

        bndbox=doc.createElement('bndbox')
        xmin=doc.createElement('xmin')
        xmin.appendChild(doc.createTextNode(str(objectlist[1])))
        ymin = doc.createElement('ymin')
        ymin.appendChild(doc.createTextNode(str(objectlist[2])))
        xmax= doc.createElement('xmax')
        xmax.appendChild(doc.createTextNode(str(objectlist[3])))
        ymax = doc.createElement('ymax')
        ymax.appendChild(doc.createTextNode(str(objectlist[4])))
        bndbox.appendChild(xmin)
        bndbox.appendChild(ymin)
        bndbox.appendChild(xmax)
        bndbox.appendChild(ymax)

        object.appendChild(name)
        object.appendChild(pose)
        object.appendChild(truncated)
        object.appendChild(difficult)
        object.appendChild(bndbox)

        annotation.appendChild(object)




    # 开始写xml文档
    fp= open(xml_dir+'.xml', 'w')
    doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")



