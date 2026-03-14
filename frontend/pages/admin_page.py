import streamlit as st
from frontend.utils.api_client import get_admin_stats, get_health


def render_admin_page():
    st.title('Admin Dashboard')
    st.caption('System health and performance metrics')

    # Health check
    health = get_health()

    if health:
        status_color = 'green' if health['status'] == 'healthy' else 'red'
        st.markdown(f'**System Status:** :{status_color}[{health["status"].upper()}]')
    else:
        st.error('Backend is not reachable. Make sure FastAPI is running.')
        return

    st.divider()

    # Stats
    try:
        stats = get_admin_stats()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Total Documents', stats['total_documents'])
        with col2:
            st.metric('Total Vectors', stats['total_vectors'])
        with col3:
            st.metric('Embedding Dimension', stats['embedding_dimension'])

        st.divider()

        st.subheader('Indexed Documents')
        if stats['documents']:
            for doc in stats['documents']:
                st.write(f'- {doc}')
        else:
            st.info('No documents indexed yet.')

        st.divider()

        # System info
        st.subheader('System Info')
        col1, col2 = st.columns(2)
        with col1:
            st.info(f'App: {health["app_name"]}')
            st.info(f'Version: {health["version"]}')
        with col2:
            st.info(f'Index loaded: {health["index_loaded"]}')
            st.info(f'Total vectors: {health["total_vectors"]}')

    except Exception as e:
        st.error(f'Could not load stats: {str(e)}')