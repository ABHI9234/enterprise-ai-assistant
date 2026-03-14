import streamlit as st
from frontend.utils.api_client import (
    stream_query, query_documents, init_session_state
)


def render_chat_page():
    st.title('Enterprise AI Knowledge Assistant')
    st.caption('Ask questions about your uploaded documents')

    init_session_state()

    # Sidebar controls
    with st.sidebar:
        st.header('Chat Settings')
        top_k = st.slider('Chunks to retrieve', 1, 10, 5)
        use_streaming = st.toggle('Streaming responses', value=True)

        if st.button('Clear chat history', use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

        st.divider()
        st.caption('Tip: Upload documents on the Upload page first')

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Chat input
    if prompt := st.chat_input('Ask a question about your documents...'):
        # Add user message
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)

        # Generate response
        with st.chat_message('assistant'):
            if use_streaming:
                response_placeholder = st.empty()
                full_response = ''
                try:
                    for token in stream_query(
                        prompt,
                        st.session_state.chat_history,
                        top_k
                    ):
                        full_response += token
                        response_placeholder.markdown(full_response + '|')
                    response_placeholder.markdown(full_response)
                except Exception as e:
                    full_response = f'Error: {str(e)}'
                    st.error(full_response)
            else:
                with st.spinner('Thinking...'):
                    try:
                        result = query_documents(
                            prompt,
                            st.session_state.chat_history,
                            top_k
                        )
                        full_response = result['answer']
                        st.markdown(full_response)
                        if result['citations']:
                            st.divider()
                            st.caption(result['citations'])
                        st.caption(f'Latency: {result["latency_ms"]}ms | Chunks used: {result["chunks_used"]}')
                    except Exception as e:
                        full_response = f'Error: {str(e)}'
                        st.error(full_response)

        # Update chat history
        st.session_state.messages.append({
            'role': 'assistant',
            'content': full_response
        })
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})
        st.session_state.chat_history.append({'role': 'assistant', 'content': full_response})