def merge(source, destination):
    """
        Merge two lists
    """
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.get(key, {})
            merge(value, node)
        else:
            destination[key] = value

    return destination