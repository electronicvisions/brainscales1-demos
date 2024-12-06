import ebrains_drive


class RepositoryInformation:
    def __init__(self, repository, name_in_the_url):
        self.repository = repository
        self.name_in_the_url = name_in_the_url

    def to_string(self):
        return "name_in_the_url=" + self.name_in_the_url + ", full name=" + self.repository.name + ", id=" + self.repository.id


def find_repository_info_from_drive_directory_path(home_path, bearer_token):
    """
    Extracts relevant information about the repository from the drive path.

    :param home_path: The home folder of the currently used Collab environment.
    :param bearer_token: The bearer token used to access the drive.
    :return: A `RepositoryInformation` object containing all relevant details about the repository.
    """
    name = home_path.replace("/mnt/user/shared/", "")
    if name.find("/") > -1:
        name = name[:name.find("/")]
    this_collabs_title = name
    ebrains_drive_client = ebrains_drive.connect(token=bearer_token)
    repo_by_title = ebrains_drive_client.repos.get_repos_by_filter("name", this_collabs_title)
    if len(repo_by_title) != 1:
        raise Exception("The repository for the collab name", this_collabs_title, "can not be found")
    # unfortunately the repository object does not return the collabs name-part-in-the-url, which is needed by the
    # quota handling (under the name of collab_id). So try to get that from the owner
    owner = repo_by_title[0].owner
    collab_name_in_the_url = owner[:owner.rindex("-")]
    # and the owner name starts with collab-, which also needs to be removed
    collab_name_in_the_url = collab_name_in_the_url[collab_name_in_the_url.find("-") + 1:]
    return RepositoryInformation(repo_by_title[0], collab_name_in_the_url)
