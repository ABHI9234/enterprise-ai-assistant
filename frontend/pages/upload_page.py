import streamlit as st
from frontend.utils.api_client import upload_document, get_documents, delete_document


def render_upload_page():
    st.title('Document Management')
    st.caption('Upload and manage your knowledge base documents')

    # Upload section
    st.subheader('Upload New Document')
    uploaded_file = st.file_uploader(
        'Choose a file',
        type=['pdf', 'docx', 'txt', 'md'],
        help='Supported formats: PDF, DOCX, TXT, Markdown'
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f'Ready to upload: {uploaded_file.name} ({uploaded_file.size:,} bytes)')
        with col2:
            if st.button('Upload & Index', type='primary', use_container_width=True):
                with st.spinner(f'Processing {uploaded_file.name}...'):
                    try:
                        result = upload_document(
                            uploaded_file.read(),
                            uploaded_file.name
                        )
                        st.success(
                            f'Successfully indexed {result["filename"]}! '
                            f'Created {result["chunks_created"]} chunks from '
                            f'{result["char_count"]:,} characters.'
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f'Upload failed: {str(e)}')

    st.divider()

    # Documents list section
    st.subheader('Indexed Documents')
    try:
        documents = get_documents()
        if not documents:
            st.info('No documents uploaded yet. Upload your first document above.')
        else:
            for doc in documents:
                with st.container(border=True):
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    with col1:
                        st.write(f'**{doc["filename"]}**')
                    with col2:
                        st.caption(doc['file_type'].upper())
                    with col3:
                        st.caption(f'{doc["chunk_count"]} chunks')
                    with col4:
                        if st.button('Delete', key=f'del_{doc["filename"]}', type='secondary'):
                            try:
                                delete_document(doc['filename'])
                                st.success(f'Deleted {doc["filename"]}')
                                st.rerun()
                            except Exception as e:
                                st.error(f'Delete failed: {str(e)}')
    except Exception as e:
        st.error(f'Could not load documents: {str(e)}')