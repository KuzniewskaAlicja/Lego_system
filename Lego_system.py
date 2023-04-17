from utils.io_handlers import OutputFile
from utils.object_description import ObjectDescriber, DescMatcher
from utils.objects_loader import ObjectsLoader


if __name__ == '__main__':
    describer = ObjectDescriber()
    out_file = OutputFile()

    objects = ObjectsLoader()()
    for image_number in range(len(objects)):
        for object_index, obj in enumerate(objects[image_number]):
            obj_info = describer(obj)
            match_idx = DescMatcher(image_number)(obj_info)
            out_file.insert(*match_idx, obj_info.circles)
    out_file.save()
